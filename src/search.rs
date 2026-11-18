//! search.rs — búsqueda híbrida: FTS5 (candidatos léxicos, ya no usa
//! bm25() como score final) + fuzzy Levenshtein (score léxico real,
//! igual que en qa_multilingue.py) + vector (semántico, vec_index.rs),
//! fusionados con TM2C2 (convex combination con normalización
//! theoretical min-max), más un boost por coincidencia de título.
//!
//! ## FTS5 + BM25 + Levenshtein
//! FTS5 genera candidatos vía OR (más recall que AND cuando la query
//! tiene 3+ palabras). BM25 rankea por rareza IDF + frecuencia en el
//! documento — "reproduce" en un título corto pesa mucho más que "como"
//! o "nunca" dispersas en un texto largo. Levenshtein sigue aportando
//! tolerancia a typos/variantes como canal de apoyo (25 %).
//!
//! ADVERTENCIA (léxico honesto): Levenshtein compara CARACTERES, no
//! PALABRAS — es ciego al orden de palabras y a paráfrasis/sinónimos.
//! "reproduce como nunca" vs. un chunk que dice "Como Nunca: cómo
//! reproducir la canción" puede dar distancia de edición alta (mala)
//! aunque el contenido sea relevante, porque compara letra por letra.
//! Por eso BM25 (que sí modela superposición de palabras) pesa 0.75
//! frente al 0.25 de Levenshtein.
//!
//! ## Por qué TM2C2 en vez de RRF
//! ...

use rusqlite::{params, Connection};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

use crate::embed::embed;
use crate::vec_index::VecIndex;

const ALPHA: f64 = 0.5;
const MIN_LEX_SCORE: f64 = 0.15; // bajado un poco: ahora f_lex es mezcla
const MIN_SEM_SCORE: f64 = 0.10;

const LEX_INF: f64 = 0.0;
const SEM_INF: f64 = -1.0;

const CONFIDENCE_Z_MIN: f64 = 1.5;

/// Vara absoluta para confiar en un candidato ÚNICO (n<2, ver abajo) —
/// más alta que `MIN_LEX_SCORE`/`MIN_SEM_SCORE` (que solo deciden si
/// un candidato entra al pool, no si hay que confiar en él como "el"
/// resultado). Sin esto, un candidato solitario en un bundle chico o
/// recién creado quedaba "confirmado" por default con solo aparecer,
/// aunque su similitud absoluta fuera mediocre — justo el caso de
/// "quien es" devolviendo un doc de instrucciones de un plugin de
/// matemática porque era el único (o casi único) chunk indexado.
const STRONG_LEX_BAR: f64 = 0.45;
const STRONG_SEM_BAR: f64 = 0.40;

/// Decide si hay un candidato "confiable" dentro de un mismo canal
/// (léxico o semántico) para normalizar los demás scores contra él.
///
/// Con 2+ candidatos: exige que se destaque estadísticamente del resto
/// (z-score ≥ `CONFIDENCE_Z_MIN`) — esto ya estaba.
///
/// Con 0 o 1 candidato: NO HAY con qué comparar, así que "ser el único"
/// no puede alcanzar por sí solo — antes sí alcanzaba (bug real de este
/// screenshot: un solo chunk en un bundle chico se normalizaba a
/// confianza plena sin importar cuán floja fuera su similitud real).
/// Ahora, para confiar en un candidato solitario, tiene que superar
/// `strong_bar` — una vara absoluta e independiente del tamaño del
/// pool. Si no la supera, se trata como si no hubiera ningún candidato
/// confiable en ese canal (mismo resultado que el `None` de abajo).
fn confident_max(scores: &[f64], strong_bar: f64) -> Option<f64> {
    let n = scores.len();
    if n < 2 {
        return scores.first().copied().filter(|&s| s >= strong_bar);
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();
    let max = scores.iter().copied().fold(f64::MIN, f64::max);
    if std < 1e-9 {
        return None;
    }
    let z = (max - mean) / std;
    // Un máximo que supera la vara ABSOLUTA también es confiable aunque no
    // se destaque estadísticamente: con consultas cortas y varios chunks
    // igual de buenos (p. ej. varios artículos de "teoría de números")
    // el z-score es bajo justamente porque HAY muchos buenos — y antes eso
    // dejaba ambos canales en "sin confianza" y la búsqueda vacía.
    if z >= CONFIDENCE_Z_MIN || max >= strong_bar {
        Some(max)
    } else {
        None
    }
}

const TITLE_EXACT_BOOST: f64 = 0.08;
const TITLE_JACCARD_WEIGHT: f64 = 0.05;
const TITLE_SUBSTRING_BOOST: f64 = 0.02;

#[derive(Serialize, Clone)]
pub struct SearchResult {
    pub id: i64,
    pub doc_id: i64,
    pub text: String,
    pub title: String,
    pub slug: String,
    pub bundle: String,
    pub okf_path: String,
    pub intent_kind: String,
    pub score: f64,
}

/// Palabras vacías (ya sin acentos, ver `fold`). Se sacan SOLO de la
/// consulta y de los títulos al comparar — nunca del índice, así que no
/// hace falta reindexar. Si la consulta es solo stopwords se usa tal cual.
const STOPWORDS: &[&str] = &[
    "y", "e", "o", "u", "la", "las", "el", "los", "lo", "un", "una", "unos", "unas", "de", "del",
    "al", "en", "a", "con", "por", "para", "que", "se", "su", "sus", "es", "son", "como", "mas",
    "the", "of", "and", "or", "an", "in", "on", "to", "for", "is", "are", "with", "by",
];

/// Minúsculas + sin marcas diacríticas ("Teoría" -> "teoria", "ñ" -> "n").
fn fold(s: &str) -> String {
    s.nfd()
        .filter(|c| !('\u{0300}'..='\u{036f}').contains(c))
        .collect::<String>()
        .to_lowercase()
}

fn split_words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

fn drop_stopwords(tokens: Vec<String>) -> Vec<String> {
    let kept: Vec<String> = tokens
        .iter()
        .filter(|t| !STOPWORDS.contains(&fold(t).as_str()))
        .cloned()
        .collect();
    if kept.is_empty() {
        tokens
    } else {
        kept
    }
}

/// Términos de la consulta (minúsculas, CON acentos, sin stopwords): es lo
/// que se manda a FTS5 y a `embed()` — el índice guarda los acentos tal
/// cual, así que acá no se pliegan.
fn query_terms(q: &str) -> Vec<String> {
    drop_stopwords(split_words(q))
}

/// Tokens "de contenido" plegados (sin acentos ni stopwords): para
/// comparar títulos y tags entre sí.
fn content_tokens(s: &str) -> Vec<String> {
    drop_stopwords(split_words(s))
        .into_iter()
        .map(|t| fold(&t))
        .collect()
}

fn title_boost(query: &str, title: &str) -> f64 {
    let q_tokens = content_tokens(query);
    let t_tokens = content_tokens(title);
    if q_tokens.is_empty() || t_tokens.is_empty() {
        return 0.0;
    }
    let q_norm = q_tokens.join(" ");
    let t_norm = t_tokens.join(" ");
    if q_norm == t_norm {
        return TITLE_EXACT_BOOST;
    }
    let q_set: FxHashSet<&String> = q_tokens.iter().collect();
    let t_set: FxHashSet<&String> = t_tokens.iter().collect();
    let overlap = q_set.intersection(&t_set).count();
    if overlap == 0 {
        return 0.0;
    }
    let union = q_set.len() + t_set.len() - overlap;
    let jaccard = overlap as f64 / union as f64;
    let substring_bonus = if t_norm.contains(&q_norm) || q_norm.contains(&t_norm) {
        TITLE_SUBSTRING_BOOST
    } else {
        0.0
    };
    TITLE_JACCARD_WEIGHT * jaccard + substring_bonus
}

/// Nuevo: boost por coincidencia de tags
fn tag_boost(query: &str, tags: &[String]) -> f64 {
    let q_tokens = content_tokens(query);
    if q_tokens.is_empty() || tags.is_empty() {
        return 0.0;
    }
    let tag_set: FxHashSet<String> = tags.iter().map(|t| fold(t)).collect();
    let matches = q_tokens
        .iter()
        .filter(|qt| tag_set.contains(qt.as_str()))
        .count();
    if matches == 0 {
        return 0.0;
    }
    0.12 * (matches as f64 / q_tokens.len() as f64)
}

/// "Camino 2" de la taxonomía Acción→Consultar/Ejecutar/Actuar: clasifica
/// la INTENCIÓN de la consulta con reglas léxicas baratas — no dispara
/// una llamada extra al LLM (eso sería un round-trip caro, y además
/// recursivo: se estaría usando el modelo para decidir cómo alimentar
/// al modelo). Los marcadores viven en un .json en disco (no acá
/// hardcodeados) para poder sumar más frases o más idiomas sin tocar ni
/// recompilar el plugin — ver `load_intent_markers` más abajo.
///
/// Los tres cubos son los mismos que en `intent_kind` de los documentos
/// (ver frontmatter.rs/db.rs): "info" (Consultar → Información),
/// "herramienta" (Ejecutar → Herramienta/Habilidad), "plan" (Actuar →
/// Plan/Proceso). El ORDEN de los grupos en el .json es el orden de
/// prioridad del match (el array `groups`, no un mapa — un mapa no
/// garantiza orden): "herramienta" y "plan" van antes que "info" porque
/// sus frases son más específicas; un match genérico de "info" nunca
/// debería tapar uno de los otros dos.
///
/// CONTRATO IMPORTANTE: si ningún marcador de ningún grupo matchea,
/// devuelve `None` — nunca hay un grupo "por defecto". Eso es a
/// propósito: un `None` que se propaga como "no aplica ningún boost de
/// intención" es preferible a forzar una clasificación adivinada, que
/// sesgaría el ranking hacia el tipo de documento equivocado y podría
/// hacer que la respuesta final use contexto que no corresponde.
pub fn classify_query_intent(query: &str) -> Option<String> {
    let q = query.to_lowercase();
    let markers = load_intent_markers();

    for group in &markers.groups {
        for phrases in group.by_lang.values() {
            if phrases.iter().any(|m| q.contains(m.as_str())) {
                return Some(group.name.clone());
            }
        }
    }
    None
}

#[derive(Deserialize)]
struct IntentMarkersFile {
    groups: Vec<IntentGroup>,
}

#[derive(Deserialize)]
struct IntentGroup {
    name: String,
    /// Cualquier clave que no sea `name` se toma como código de idioma
    /// ("es", "en", ...) con su lista de frases — así se agregan idiomas
    /// nuevos sin cambiar la forma del archivo, solo agregando una clave.
    #[serde(flatten)]
    by_lang: FxHashMap<String, Vec<String>>,
}

/// El .json de marcadores NO está embebido en el binario — vive
/// únicamente en disco (`Knowledge/intent_markers.json`). Si no existe
/// o está roto, `load_intent_markers` devuelve una lista de grupos
/// vacía: `classify_query_intent` simplemente no clasifica nada (`None`
/// siempre), degradado en vez de roto, y el plugin sigue funcionando
/// con búsqueda léxica+semántica normal, sin el boost de intención.
///
/// Carga y parsea el .json DESDE DISCO en cada llamada (no se cachea en
/// memoria): el archivo es chico, esto se llama una vez por consulta
/// (no en un hot loop), y así una edición manual se aplica de inmediato,
/// sin reiniciar A.M.E.
fn load_intent_markers() -> IntentMarkersFile {
    let path = crate::knowledge_dir().join("intent_markers.json");

    match std::fs::read_to_string(&path) {
        Ok(raw) => match serde_json::from_str::<IntentMarkersFile>(&raw) {
            Ok(parsed) => parsed,
            Err(e) => {
                log_bad_markers_once(&format!(
                    "intent_markers.json inválido ({e}) — sin clasificación de intención hasta que se corrija"
                ));
                IntentMarkersFile { groups: vec![] }
            }
        },
        Err(_) => {
            log_bad_markers_once(&format!(
                "no existe {} — sin clasificación de intención (creá el archivo para habilitarla)",
                path.display()
            ));
            IntentMarkersFile { groups: vec![] }
        }
    }
}

/// Evita inundar el log en cada mensaje si el archivo falta o está
/// roto: `on_before_request` llama a esto una vez por mensaje del
/// usuario — sin este guard, se repetiría la misma advertencia todo
/// el rato que dure la sesión.
fn log_bad_markers_once(msg: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    if WARNED.set(()).is_ok() {
        crate::plugin_log(&format!("WARN {msg}"));
    }
}

/// Vara MÁS ALTA que la de `MIN_LEX_SCORE`/`MIN_SEM_SCORE`, pensada
/// específicamente para lo que se INYECTA en el prompt del LLM
/// (`on_before_request`), no para lo que se muestra en la pestaña de
/// Buscar. La diferencia importa: un resultado de búsqueda borderline
/// lo filtra un humano con la vista; un resultado borderline inyectado
/// como "contexto recuperado" el modelo lo va a tratar como un hecho
/// confiable y puede terminar citándolo — mejor no mandar nada que
/// mandar algo dudoso presentado con la autoridad de "esto es lo que
/// sabemos". Valor de partida conservador; ajustar mirando los scores
/// que loguea `cmd_on_before_request` en casos reales de uso.
pub const INJECT_MIN_SCORE: f64 = 0.20;

const INTENT_BOOST: f64 = 0.10;

/// Boost cuando la intención detectada en la consulta coincide con el
/// `intent_kind` que se le asignó al documento al ingestarlo ("camino 1").
/// Si la consulta no clasificó en ningún cubo, o el documento no tiene
/// `intent_kind` seteado, no hay boost — degradado, no roto: un doc sin
/// clasificar sigue compitiendo solo por léxico+semántico, como antes.
fn intent_boost(query_intent: Option<&str>, doc_intent: &str) -> f64 {
    match query_intent {
        Some(qi) if qi == doc_intent => INTENT_BOOST,
        _ => 0.0,
    }
}

/// FTS5 con MATCH: OR explícito entre tokens.  AND era más rápido pero
/// descarta documentos que no contienen TODAS las palabras (ej. un
/// plugin de música con "reproduce" pero sin "como" o "nunca").  OR
/// devuelve más candidatos; BM25 se encarga de penalizar los que solo
/// tienen palabras comunes sueltas.
fn fts_query(q: &str) -> String {
    let tokens: Vec<String> = query_terms(q)
        .into_iter()
        .map(|s| format!("\"{s}\""))
        .collect();
    if tokens.is_empty() {
        "\"\"".to_string()
    } else {
        tokens.join(" OR ")
    }
}

/// Devuelve candidatos léxicos + su score BM25 crudo.  BM25 en FTS5
/// devuelve valores negativos (más negativo = más relevante).
fn lexical_search(
    conn: &Connection,
    query: &str,
    bundle: Option<&str>,
    limit: usize,
) -> Vec<(i64, f64)> {
    let q = fts_query(query);
    let sql = if bundle.is_some() {
        "SELECT c.rowid, bm25(f) FROM chunks_fts f
         JOIN chunks c ON c.id = f.rowid
         JOIN documents d ON d.id = c.doc_id
         WHERE f.chunks_fts MATCH ?1 AND d.bundle = ?2
         ORDER BY bm25(f) LIMIT ?3"
    } else {
        "SELECT rowid, bm25(chunks_fts) FROM chunks_fts
         WHERE chunks_fts MATCH ?1
         ORDER BY bm25(chunks_fts) LIMIT ?2"
    };
    let Ok(mut stmt) = conn.prepare(sql) else {
        return vec![];
    };
    if let Some(b) = bundle {
        stmt.query_map(params![q, b, limit as i64], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
        })
        .map(|it| it.filter_map(Result::ok).collect())
        .unwrap_or_default()
    } else {
        stmt.query_map(params![q, limit as i64], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
        })
        .map(|it| it.filter_map(Result::ok).collect())
        .unwrap_or_default()
    }
}

fn fuzzy_lex_score(query: &str, title: &str, body: &str) -> f64 {
    let q_tokens = crate::qa::universal_tokenize(query);
    if q_tokens.is_empty() {
        return 0.0;
    }

    let mut snippet = title.to_string();
    snippet.push(' ');
    snippet.push_str(
        &crate::qa::split_sentences(body)
            .into_iter()
            .take(3)
            .collect::<Vec<_>>()
            .join(" "),
    );
    let doc_tokens = crate::qa::universal_tokenize(&snippet);
    if doc_tokens.is_empty() {
        return 0.0;
    }

    let per_query_token_best: Vec<f64> = q_tokens
        .iter()
        .map(|qt| {
            doc_tokens
                .iter()
                .map(|dt| crate::qa::levenshtein_similarity(qt, dt, 40))
                .fold(0.0, f64::max)
        })
        .collect();

    per_query_token_best.iter().sum::<f64>() / per_query_token_best.len() as f64
}

fn semantic_search(
    vec_index: &VecIndex,
    query: &str,
    bundle: Option<&str>,
    limit: usize,
) -> Vec<(i64, f64)> {
    // Sin stopwords: en el hashing vectorizer cada palabra pesa 1.0, así
    // que "y la de" dominaban el vector y llenaban el top-k de chunks
    // llenos de palabras comunes.
    let q = embed(&query_terms(query).join(" "));
    // k generoso también SIN filtro de bundle (antes era `limit` = 30 sobre
    // todo el corpus): un chunk que queda fuera del top-k semántico recibe
    // f_sem = -1 y el filtro duro lo descartaba.
    let k = (limit * 4).max(100);
    vec_index
        .search(bundle, &q, k)
        .into_iter()
        .map(|(id, dist)| (id, 1.0 - dist as f64))
        .collect()
}

fn theoretical_min_max(f: f64, inf: f64, m_q: f64) -> f64 {
    let denom = m_q - inf;
    if denom <= 0.0 {
        0.0
    } else {
        ((f - inf) / denom).clamp(0.0, 1.0)
    }
}

pub fn hybrid_search(
    conn: &Connection,
    vec_index: &VecIndex,
    query: &str,
    bundle: Option<&str>,
    top_k: usize,
) -> Vec<SearchResult> {
    let lex_results = lexical_search(conn, query, bundle, 30);
    let sem_results = semantic_search(vec_index, query, bundle, 30);
    let sem_map: FxHashMap<i64, f64> = sem_results.into_iter().collect();

    // ── Normalizar BM25 a [0,1] (invertido: más negativo = mejor = 1.0) ──
    let bm25_map: FxHashMap<i64, f64> = lex_results.iter().copied().collect();
    let bm25_vals: Vec<f64> = lex_results.iter().map(|(_, s)| *s).collect();
    let (bm25_min, bm25_max) = bm25_vals
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
            (min.min(v), max.max(v))
        });
    let bm25_norm = |raw: f64| -> f64 {
        if bm25_max > bm25_min {
            // bm25 más negativo → 1.0, menos negativo → 0.0
            (bm25_max - raw) / (bm25_max - bm25_min)
        } else if bm25_max.is_finite() {
            1.0
        } else {
            0.0
        }
    };

    let mut cids: FxHashSet<i64> = FxHashSet::default();
    cids.extend(lex_results.iter().map(|(id, _)| *id));
    cids.extend(sem_map.keys().copied());

    // ── Bonus: canal de título ──
    // Todos los términos de contenido deben estar en el título. LIKE de
    // SQLite no ignora acentos, así que en tokens de 4+ letras cada vocal
    // pasa a `_` (comodín de 1 carácter): "teoria" matchea "Teoría". Solo
    // AGREGA candidatos (siguen pasando por los filtros de score), un
    // candidato por documento y los títulos más cortos primero.
    let title_candidates: Vec<i64> = {
        let tokens = content_tokens(query);
        if tokens.is_empty() {
            vec![]
        } else {
            let likes = tokens
                .iter()
                .map(|t| {
                    let pat: String = if t.chars().count() >= 4 {
                        t.chars()
                            .map(|c| if "aeiou".contains(c) { '_' } else { c })
                            .collect()
                    } else {
                        t.clone()
                    };
                    format!("LOWER(d.title) LIKE '%{}%'", pat.replace('\'', "''"))
                })
                .collect::<Vec<_>>()
                .join(" AND ");
            let bundle_filter = if let Some(b) = bundle {
                format!(" AND d.bundle = '{}'", b.replace('\'', "''"))
            } else {
                String::new()
            };
            let sql = format!(
                "SELECT MIN(c.id) FROM chunks c JOIN documents d ON d.id = c.doc_id \
                 WHERE {likes}{bundle_filter} GROUP BY d.id ORDER BY length(d.title) LIMIT 10"
            );
            conn.prepare(&sql)
                .ok()
                .and_then(|mut stmt| {
                    stmt.query_map([], |r| r.get::<_, i64>(0))
                        .ok()
                        .map(|rows| rows.filter_map(Result::ok).collect::<Vec<i64>>())
                })
                .unwrap_or_default()
        }
    };
    cids.extend(title_candidates);

    if cids.is_empty() {
        return vec![];
    }

    let sem_scores: Vec<f64> = sem_map.values().copied().collect();
    let m_sem = confident_max(&sem_scores, STRONG_SEM_BAR).unwrap_or(SEM_INF);

    let cids_vec: Vec<i64> = cids.into_iter().collect();
    let placeholders: String = cids_vec.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT c.id, c.doc_id, c.text, d.title, d.slug, d.bundle, d.okf_path, d.tags, d.intent_kind
         FROM chunks c JOIN documents d ON d.id = c.doc_id
         WHERE c.id IN ({placeholders})"
    );
    let mut rows_by_id: FxHashMap<
        i64,
        (i64, String, String, String, String, String, String, String),
    > = FxHashMap::default();
    if let Ok(mut stmt) = conn.prepare(&sql) {
        let params: Vec<&dyn rusqlite::ToSql> =
            cids_vec.iter().map(|c| c as &dyn rusqlite::ToSql).collect();
        if let Ok(mut rows) = stmt.query(params.as_slice()) {
            while let Ok(Some(r)) = rows.next() {
                if let (
                    Ok(id),
                    Ok(doc_id),
                    Ok(text),
                    Ok(title),
                    Ok(slug),
                    Ok(bundle),
                    Ok(okf_path),
                    Ok(tags_json),
                    Ok(intent_kind),
                ) = (
                    r.get::<_, i64>(0),
                    r.get::<_, i64>(1),
                    r.get::<_, String>(2),
                    r.get::<_, String>(3),
                    r.get::<_, String>(4),
                    r.get::<_, String>(5),
                    r.get::<_, String>(6),
                    r.get::<_, String>(7),
                    r.get::<_, String>(8),
                ) {
                    rows_by_id.insert(
                        id,
                        (
                            doc_id,
                            text,
                            title,
                            slug,
                            bundle,
                            okf_path,
                            tags_json,
                            intent_kind,
                        ),
                    );
                }
            }
        }
    }

    // ── f_lex combinado: 75 % BM25 + 25 % Levenshtein ──
    let mut lex_combined: FxHashMap<i64, f64> = FxHashMap::default();
    for (cid, (_, text, title, _, _, _, _, _)) in rows_by_id.iter() {
        let lev = fuzzy_lex_score(query, title, text);
        // Sin hit de BM25 (llegó por el canal semántico o de título): 0.0.
        // Antes se normalizaba f64::MAX, que daba un score léxico
        // astronómicamente negativo.
        let bm25_n = bm25_map.get(cid).map(|&raw| bm25_norm(raw)).unwrap_or(0.0);
        let combined = 0.75 * bm25_n + 0.25 * lev;
        lex_combined.insert(*cid, combined);
    }

    let lex_scores: Vec<f64> = lex_combined.values().copied().collect();
    let m_lex = confident_max(&lex_scores, STRONG_LEX_BAR).unwrap_or(LEX_INF);

    // "Camino 2": clasificamos la intención de la consulta UNA sola vez
    // acá afuera del loop — es una consulta, no un doc por doc.
    let query_intent = classify_query_intent(query);
    let mut resolved: Vec<SearchResult> = vec![];
    for (cid, (doc_id, text, title, slug, doc_bundle, okf_path, tags_json, intent_kind)) in
        rows_by_id
    {
        // ── Filtro duro por intención (a pedido, más estricto que el
        // boost de abajo) ──
        // Si la consulta clasificó en un cubo ("info"/"herramienta"/
        // "plan") Y el documento tiene un `intent_kind` seteado, y NO
        // coinciden, se descarta directo — no compite ni por léxico ni
        // por semántico. Es justamente el caso de este screenshot:
        // "quien es" clasifica como "info", pero el único contenido
        // indexado son instrucciones de un plugin tageadas "herramienta".
        //
        // Dos guards a propósito, para no over-filtrar:
        // - Si `query_intent` es `None` (la consulta no matcheó ningún
        //   marcador), NO se filtra nada acá — no hay de qué lado
        //   comparar, y ya vimos que forzar una clasificación adivinada
        //   es peor que no clasificar.
        // - Si el documento NO tiene `intent_kind` (nunca se lo
        //   asignaron), tampoco se filtra — un doc sin clasificar sigue
        //   compitiendo solo por léxico+semántico, como si este filtro
        //   no existiera. Si se excluyera también a los sin clasificar,
        //   CUALQUIER búsqueda con intención detectada perdería todo el
        //   contenido viejo/sin taggear — un efecto colateral mucho peor
        //   que el problema que se quiere resolver.
        //
        // Trade-off a tener en cuenta: el clasificador de intención es
        // heurístico (matching de frases, no un LLM) — un falso negativo
        // acá (la consulta clasifica mal, o el documento está mal
        // tageado) hace que un resultado genuinamente relevante quede
        // afuera SIN posibilidad de rescate por score. Con el boost
        // (como estaba antes) ese mismo caso simplemente competía en
        // igualdad de condiciones; con este filtro, directamente no
        // aparece. Si notás documentos relevantes desapareciendo de la
        // búsqueda, la causa más probable es un `intent_kind` mal puesto
        // en el ingest, no un bug de search.
        if let Some(qi) = query_intent.as_deref() {
            if !intent_kind.is_empty() && intent_kind != qi {
                continue;
            }
        }

        let f_lex = lex_combined.get(&cid).copied().unwrap_or(LEX_INF);
        let f_sem = sem_map.get(&cid).copied().unwrap_or(SEM_INF);

        // Basta UNA señal suficiente (léxica O semántica). Antes exigía las
        // dos, así que un chunk perfecto por BM25 que no entraba al top-k
        // vectorial (f_sem = -1) desaparecía. `phi_*` más abajo ya pone en
        // 0 el canal ausente, y los boosts nunca fabrican señal.
        if f_lex < MIN_LEX_SCORE && f_sem < MIN_SEM_SCORE {
            continue;
        }

        let phi_lex = theoretical_min_max(f_lex, LEX_INF, m_lex);
        let phi_sem = theoretical_min_max(f_sem, SEM_INF, m_sem);

        // ── Fix real del "me devolvió algo sin nada que ver" ──
        // `phi_lex`/`phi_sem` dan 0.0 en dos casos: (a) este candidato
        // puntual es el peor de su canal, o (b) NINGÚN candidato de la
        // consulta se destacó con confianza estadística en ese canal
        // (`confident_max` devolvió `None` — ver arriba). En cualquiera
        // de los dos casos, 0.0 significa "sin señal real de este
        // canal" — y si los DOS canales dan 0.0, este resultado no
        // tiene ninguna evidencia léxica ni semántica de ser relevante.
        // Antes, un boost de tag/título/intención (que puede dispararse
        // por una sola palabra suelta en común) alcanzaba para inflar
        // ese 0.0 a un score positivo y meter el doc en el top — un
        // boost tiene que AMPLIFICAR una señal que ya existe, nunca
        // fabricar una de la nada. Esto es lo que probablemente estaba
        // devolviendo contexto sin relación real con la consulta.
        if phi_lex <= 0.0 && phi_sem <= 0.0 {
            continue;
        }

        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        let score = ALPHA * phi_sem
            + (1.0 - ALPHA) * phi_lex
            + title_boost(query, &title)
            + tag_boost(query, &tags)
            + intent_boost(query_intent.as_deref(), &intent_kind);
        resolved.push(SearchResult {
            id: cid,
            doc_id,
            text,
            title,
            slug,
            bundle: doc_bundle,
            okf_path,
            intent_kind,
            score,
        });
    }
    resolved.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    resolved.truncate(top_k);
    resolved
}