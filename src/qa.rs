//! qa.rs — puerto de `qa_multilingue.py` a Rust.
//!
//! Se usa DESPUÉS de reagrupar todos los chunks de un documento
//! (`db::document_full_text`) y ANTES de inyectarlo al contexto del LLM
//! en `cmd_on_before_request`: en vez de mandar el documento entero
//! (que puede pesar decenas de miles de caracteres si el .md original
//! era largo), lo particiona en secciones, las rankea contra la query
//! del usuario con las mismas métricas que `qa_multilingue.py`
//! (coseno TF-IDF 0.55 + Jaccard 0.30 + Levenshtein 0.15) y devuelve
//! solo las secciones más relevantes hasta un presupuesto de
//! caracteres — resolviendo el mismo problema de overflow de contexto
//! que ataca el paper de IBM (Labate et al., 2025), pero para texto
//! recuperado en vez de outputs de tools.
//!
//! Fórmulas y pesos calcados 1:1 del script original para no cambiar
//! el comportamiento ya validado ahí.

use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::OnceLock;

// ------------------------
// Regex compiladas una sola vez (equivalente a los `re.compile` a nivel
// módulo de Python).
// ------------------------
fn re_cjk() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"[\u{4e00}-\u{9fff}\u{3040}-\u{309f}\u{30a0}-\u{30ff}\u{ac00}-\u{d7af}\u{0e00}-\u{0e7f}]").unwrap()
    })
}
fn re_word() -> &'static Regex {
    // Python: r'[^\W_]+' con re.UNICODE — "letras/dígitos unicode, sin '_'".
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[^\W_]+").unwrap())
}
fn re_sep() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[=\-*_]{3,}$").unwrap())
}
fn re_meta_kv() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*:").unwrap())
}
fn re_header() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^(#{1,6})\s+(.*)").unwrap())
}
fn re_paren_num() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^\(\d+\)").unwrap())
}
fn re_img() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"!\[.*?\]\(.*?\)").unwrap())
}
fn re_link() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\[([^\]]+)\]\(.*?\)").unwrap())
}
fn re_emph() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[*_]{1,3}(.*?)[*_]{1,3}").unwrap())
}
fn re_ws() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\s+").unwrap())
}
fn re_sentence_split() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // Python usa lookbehind `(?<=[.!?])\s+`; el crate `regex` no soporta
    // lookbehind, así que partimos a mano en split_sentences().
    R.get_or_init(|| Regex::new(r"[.!?]\s+").unwrap())
}

fn universal_q_words() -> &'static FxHashSet<&'static str> {
    static W: OnceLock<FxHashSet<&'static str>> = OnceLock::new();
    W.get_or_init(|| {
        [
            "quien", "quienes", "que", "cual", "cuales", "como", "donde", "cuando", "por", "who",
            "what", "which", "how", "where", "when", "why", "qui", "quel", "comment", "ou",
            "quand", "pourquoi", "quem", "qual", "onde", "quando", "chi", "cosa", "quale", "dove",
            "perche",
        ]
        .into_iter()
        .collect()
    })
}

// ------------------------
// Estructura
// ------------------------
#[derive(Clone)]
pub struct QaChunk {
    pub heading: String,
    pub body: String,
    pub text: String,
    pub tokens: Vec<String>,
}

// ------------------------
// 1. Normalización multilingüe
// ------------------------
// NFKC preserva equivalencias de compatibilidad (ligaduras, ancho completo
// CJK, etc.) antes de bajar a minúsculas para que la tokenización sea estable.
fn normalize(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkc().collect::<String>().to_lowercase()
}

pub(crate) fn universal_tokenize(text: &str) -> Vec<String> {
    let norm = normalize(text);
    let with_cjk_spaced = re_cjk().replace_all(&norm, " $0 ").to_string();

    let mut out = Vec::new();
    for m in re_word().find_iter(&with_cjk_spaced) {
        let t = m.as_str();
        if t.chars().all(|c| c.is_ascii_digit() || c.is_numeric()) {
            continue;
        }
        let is_cjk_single = t.chars().count() == 1 && re_cjk().is_match(t);
        if t.chars().count() < 2 && !is_cjk_single {
            continue;
        }
        out.push(t.to_string());
    }
    out
}

pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let collapsed = re_ws().replace_all(text.trim(), " ").to_string();
    if collapsed.is_empty() {
        return vec![];
    }
    // Partimos manteniendo el delimitador pegado a la oración anterior,
    // igual que el lookbehind de Python.
    let mut sentences = Vec::new();
    let mut last = 0;
    for m in re_sentence_split().find_iter(&collapsed) {
        let end = m.start() + 1; // incluye el . ! ?
        sentences.push(collapsed[last..end].trim().to_string());
        last = m.end();
    }
    if last < collapsed.len() {
        sentences.push(collapsed[last..].trim().to_string());
    }
    sentences.into_iter().filter(|s| !s.is_empty()).collect()
}

// ------------------------
// 2. Extracción por secciones
// ------------------------
pub fn extract_candidates(file_text: &str) -> Vec<QaChunk> {
    let text = re_img().replace_all(file_text, "");
    let text = re_link().replace_all(&text, "$1");
    let text = re_emph().replace_all(&text, "$1");

    const SEP: &str = "\u{0}__SECTION_SEPARATOR__\u{0}";
    let mut cleaned_lines: Vec<String> = Vec::new();

    for line in text.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if re_sep().is_match(s) {
            cleaned_lines.push(SEP.to_string());
            continue;
        }
        if re_meta_kv().is_match(s) {
            continue;
        }
        if s.starts_with("Artículo de Wikipedia")
            || s.starts_with("Fuente:")
            || s.starts_with("Título:")
            || s.starts_with("Tags:")
            || s.starts_with("Datos:")
            || s.starts_with("```")
        {
            continue;
        }
        if s.starts_with('•') || s.starts_with('-') || s.starts_with('–') {
            continue;
        }
        cleaned_lines.push(s.to_string());
    }

    let joined = cleaned_lines.join("\n");
    let raw_chunks: Vec<&str> = joined
        .split(SEP)
        .map(|c| c.trim())
        .filter(|c| !c.is_empty())
        .collect();

    let mut candidates = Vec::new();
    let mut previous_title = "Documento".to_string();

    for chunk in raw_chunks {
        let lines: Vec<&str> = chunk
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if lines.is_empty() {
            continue;
        }

        let mut first = lines[0].to_string();
        let word_count = |s: &str| s.split_whitespace().count();

        let mut is_section_title = (first.chars().all(|c| !c.is_lowercase())
            && word_count(&first) <= 8
            && first.chars().count() > 3)
            || (re_paren_num().is_match(&first) && word_count(&first) <= 6);

        if let Some(caps) = re_header().captures(&first) {
            is_section_title = true;
            first = caps.get(2).unwrap().as_str().to_string();
        }

        let (heading, body) = if is_section_title {
            (first.clone(), lines[1..].join(" ").trim().to_string())
        } else {
            (previous_title.clone(), lines.join(" ").trim().to_string())
        };

        let heading_upper = heading.to_uppercase();
        if [
            "REFERENCIAS",
            "VÉASE TAMBIÉN",
            "ENLACES EXTERNOS",
            "REFERENCES",
            "SEE ALSO",
        ]
        .contains(&heading_upper.as_str())
        {
            continue;
        }

        let body = if body.is_empty() {
            heading.clone()
        } else {
            body
        };
        let body = re_ws().replace_all(&body, " ").trim().to_string();
        let text_for_index = format!("{heading}. {body}").trim().to_string();
        let tokens = universal_tokenize(&text_for_index);

        if tokens.len() >= 2 {
            candidates.push(QaChunk {
                heading: heading.clone(),
                body,
                text: text_for_index,
                tokens,
            });
        }

        if is_section_title {
            previous_title = heading;
        }
    }

    candidates
}

// ------------------------
// 3. TF-IDF y métricas (fórmulas exactas de qa_similitudes)
// ------------------------
fn build_tfidf(
    corpus_tokens: &[Vec<String>],
) -> (Vec<FxHashMap<String, f64>>, FxHashMap<String, f64>) {
    let n_docs = corpus_tokens.len() as f64;
    let mut df: FxHashMap<&str, usize> = FxHashMap::default();
    for tokens in corpus_tokens {
        let uniq: FxHashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        for t in uniq {
            *df.entry(t).or_insert(0) += 1;
        }
    }

    let idf: FxHashMap<String, f64> = df
        .iter()
        .map(|(term, &d)| {
            let v = ((1.0 + n_docs) / (1.0 + d as f64)).ln() + 1.0;
            (term.to_string(), v)
        })
        .collect();

    let mut vectors = Vec::with_capacity(corpus_tokens.len());
    for tokens in corpus_tokens {
        let mut tf: FxHashMap<&str, usize> = FxHashMap::default();
        for t in tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        let total = tokens.len().max(1) as f64;
        let vec: FxHashMap<String, f64> = tf
            .into_iter()
            .map(|(term, count)| {
                let w = (count as f64 / total) * idf.get(term).copied().unwrap_or(0.0);
                (term.to_string(), w)
            })
            .collect();
        vectors.push(vec);
    }

    (vectors, idf)
}

fn cosine_similarity(a: &FxHashMap<String, f64>, b: &FxHashMap<String, f64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .filter_map(|(t, w)| b.get(t).map(|wb| w * wb))
        .sum();
    if dot == 0.0 {
        return 0.0;
    }
    let norm_a = a.values().map(|w| w * w).sum::<f64>().sqrt();
    let norm_b = b.values().map(|w| w * w).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    let set_a: FxHashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: FxHashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }
    let inter = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;
    inter / union
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    let (a_chars, b_chars): (Vec<char>, Vec<char>) = if a.chars().count() < b.chars().count() {
        (b.chars().collect(), a.chars().collect())
    } else {
        (a.chars().collect(), b.chars().collect())
    };
    let mut previous: Vec<usize> = (0..=b_chars.len()).collect();
    for (i, &ca) in a_chars.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, &cb) in b_chars.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            let ins_del = (previous[j + 1] + 1).min(current[j] + 1);
            current.push(ins_del.min(previous[j] + cost));
        }
        previous = current;
    }
    *previous.last().unwrap()
}

pub(crate) fn levenshtein_similarity(a: &str, b: &str, max_chars: usize) -> f64 {
    let a = normalize(a);
    let mut b = normalize(b);
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    b = b.chars().take(max_chars).collect();
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 0.0;
    }
    1.0 - (levenshtein_distance(&a, &b) as f64 / max_len as f64)
}

fn best_levenshtein_for_candidate(query: &str, chunk: &QaChunk) -> f64 {
    let mut snippets = vec![chunk.heading.clone()];
    snippets.extend(split_sentences(&chunk.body).into_iter().take(3));
    snippets
        .iter()
        .map(|snippet| {
            levenshtein_similarity(query, &format!("{} {}", chunk.heading, snippet), 180)
        })
        .fold(0.0, f64::max)
}

// ------------------------
// 4. Ranking
// ------------------------
pub struct RankedSection {
    pub heading: String,
    pub body: String,
    pub score: f64,
}

fn rank_candidates(
    query: &str,
    candidates: &[QaChunk],
    tfidf_vectors: &[FxHashMap<String, f64>],
    idf: &FxHashMap<String, f64>,
) -> Vec<RankedSection> {
    let query_tokens = universal_tokenize(query);
    let mut tf: FxHashMap<&str, usize> = FxHashMap::default();
    for t in &query_tokens {
        *tf.entry(t.as_str()).or_insert(0) += 1;
    }
    let total = query_tokens.len().max(1) as f64;
    let query_vector: FxHashMap<String, f64> = tf
        .into_iter()
        .filter_map(|(term, count)| {
            idf.get(term)
                .map(|&i| (term.to_string(), (count as f64 / total) * i))
        })
        .collect();

    const W_COS: f64 = 0.55;
    const W_JAC: f64 = 0.30;
    const W_LEV: f64 = 0.15;

    let mut results: Vec<RankedSection> = candidates
        .iter()
        .enumerate()
        .map(|(idx, chunk)| {
            let cos = cosine_similarity(&query_vector, &tfidf_vectors[idx]);
            let jac = jaccard_similarity(&query_tokens, &chunk.tokens);
            let lev = best_levenshtein_for_candidate(query, chunk);
            let score = W_COS * cos + W_JAC * jac + W_LEV * lev;
            RankedSection {
                heading: chunk.heading.clone(),
                body: chunk.body.clone(),
                score,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ------------------------
// 5. Condensación para inyección de contexto
// ------------------------
// Reemplaza a `compose_answer` de qa_multilingue.py (que devolvía texto
// listo para responder). Acá el objetivo es distinto: no queremos UNA
// respuesta, queremos las secciones más relevantes del documento hasta
// llenar un presupuesto de caracteres — el mecanismo que evita el
// overflow de contexto cuando el documento reagrupado es enorme.
//
// Igual que el paper de IBM: si el documento entero ya entra dentro del
// presupuesto, se devuelve completo sin tocar nada (no interfiere en
// los casos donde no hace falta condensar).
//
// Importante: este límite es un contrato de seguridad para el consumidor
// (Agentic). No basta con elegir *secciones* cuyo tamaño estimado entre en
// el presupuesto: un PDF mal extraído puede llegar como una única sección
// gigantesca. Toda salida de esta función debe respetarlo, incluso en ese
// caso.
fn limitar_caracteres(texto: &str, limite: usize) -> String {
    if texto.chars().count() <= limite {
        return texto.to_string();
    }

    const AVISO: &str = "\n[... documento truncado por tamaño ...]";
    let aviso_len = AVISO.chars().count();
    if limite <= aviso_len {
        return AVISO.chars().take(limite).collect();
    }

    let mut salida: String = texto.chars().take(limite - aviso_len).collect();
    salida.push_str(AVISO);
    salida
}

pub fn condense(query: &str, doc_text: &str, budget_chars: usize) -> String {
    if doc_text.chars().count() <= budget_chars {
        return doc_text.to_string();
    }

    let candidates = extract_candidates(doc_text);
    if candidates.is_empty() {
        // No se pudo seccionar (documento sin estructura reconocible):
        // fallback a un corte duro con aviso, mejor que mandar todo.
        return limitar_caracteres(doc_text, budget_chars);
    }

    let corpus_tokens: Vec<Vec<String>> = candidates.iter().map(|c| c.tokens.clone()).collect();
    let (tfidf_vectors, idf) = build_tfidf(&corpus_tokens);
    let ranked = rank_candidates(query, &candidates, &tfidf_vectors, &idf);

    // Igual que compose_answer: si la query es una pregunta, conviene
    // margen para combinar más de una sección — acá no limitamos a 2,
    // sino a lo que entre en el presupuesto, tomando en orden de score.
    let is_question = query.contains('?')
        || universal_tokenize(query)
            .iter()
            .any(|t| universal_q_words().contains(t.as_str()));

    let mut selected: Vec<&RankedSection> = Vec::new();
    let mut used = 0usize;
    let best_score = ranked.first().map(|r| r.score).unwrap_or(0.0);

    for r in &ranked {
        // Sin ser pregunta, nos quedamos solo con lo mejor rankeado
        // salvo que quepa más dentro del presupuesto igual.
        if !is_question && !selected.is_empty() && r.score < best_score * 0.65 {
            break;
        }
        let piece_len = r.heading.chars().count() + r.body.chars().count() + 4;
        if used + piece_len > budget_chars && !selected.is_empty() {
            break;
        }
        selected.push(r);
        used += piece_len;
        if used >= budget_chars {
            break;
        }
    }

    if selected.is_empty() {
        selected.push(&ranked[0]);
    }

    let mut out = selected
        .iter()
        .map(|s| format!("## {}\n{}", s.heading, s.body))
        .collect::<Vec<_>>()
        .join("\n\n");

    if selected.len() < candidates.len() {
        out.push_str("\n[... resto del documento omitido por tamaño ...]");
    }

    // `selected` puede contener una primera sección más grande que todo el
    // presupuesto (un PDF convertido en un solo bloque). El ranking debe
    // conservar la sección más relevante, pero nunca puede romper el límite
    // que protege la ventana de contexto del modelo.
    limitar_caracteres(&out, budget_chars)
}

#[cfg(test)]
mod tests {
    use super::condense;

    #[test]
    fn condense_never_exceeds_budget_for_one_huge_section() {
        // Simula una extracción de PDF sin saltos de sección útiles: antes
        // este caso elegía el único candidato entero y devolvía megabytes.
        let document = format!("## Canto primero\n{}", "verso extenso ".repeat(10_000));
        let condensed = condense("Canto primero", &document, 320);

        assert!(condensed.chars().count() <= 320);
        assert!(condensed.contains("Canto primero"));
    }
}
