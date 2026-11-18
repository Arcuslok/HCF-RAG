//! graph.rs — grafo de relaciones (links explícitas + tags compartidos +
//! similitud de embeddings opcional) y mapa 2D de similitud (PCA vía power
//! iteration). Ambos son bajo demanda desde la UI.
//!
//! Diseño del grafo (v2): un grafo legible tiene POCAS aristas por nodo y
//! pocos nodos. Por eso:
//!   * las aristas se eligen con un tope SIMÉTRICO por nodo (un hub no
//!     puede acumular grado ilimitado),
//!   * los tags muy comunes no generan aristas (no son señal) y cada par
//!     emite UNA sola arista aunque comparta varios tags,
//!   * la similitud por embeddings es opcional (`GraphOpts::sim`) y usa
//!     vecinos más cercanos sobre vectores CENTRADOS — con vectores de
//!     hashing no centrados casi todos los pares superan cualquier umbral,
//!   * si hay más documentos que `max_nodes` se muestran los más
//!     conectados (el navegador no puede dibujar miles de nodos).

use rusqlite::{params, Connection};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;

const TAG_EDGE_WEIGHT: f64 = 0.5;
const LINK_EDGE_WEIGHT: f64 = 0.85;
/// Coseno mínimo (sobre centroides centrados) para aceptar un vecino.
const SIM_EDGE_MIN: f32 = 0.10;
/// Vecinos más cercanos que propone cada nodo en el canal de similitud.
const SIM_NEIGHBORS: usize = 4;
/// Tope de aristas por nodo, aplicado a AMBOS extremos.
const MAX_EDGES_PER_NODE: usize = 6;
/// Un tag compartido por más documentos que esto no es señal útil.
const TAG_MAX_DOCS: usize = 40;
/// Nodos que se envían al navegador por defecto.
pub const DEFAULT_MAX_NODES: usize = 400;
/// Tope de documentos que se cargan/analizan (links y tags son baratos).
const MAX_DOCS_FOR_PAIRWISE_SIM: usize = 5000;

#[derive(Clone, Copy)]
pub struct GraphOpts {
    /// Calcular aristas por similitud de embeddings (O(n²) sobre los
    /// nodos ya recortados a `max_nodes`).
    pub sim: bool,
    pub max_nodes: usize,
}

impl Default for GraphOpts {
    fn default() -> Self {
        GraphOpts {
            sim: false,
            max_nodes: DEFAULT_MAX_NODES,
        }
    }
}

#[derive(Serialize)]
pub struct GraphNode {
    pub id: i64,
    pub slug: String,
    pub bundle: String,
    pub bundle_idx: usize,
    pub title: String,
    pub source_kind: String,
    pub tags: Vec<String>,
    pub links: Vec<String>,
    pub chunks: i64,
    pub degree: usize,
}

#[derive(Serialize, Clone)]
pub struct Edge {
    source: i64,
    target: i64,
    weight: f64,
    #[serde(rename = "type")]
    kind: &'static str,
    priority: u8,
}

#[derive(Serialize)]
pub struct OkfGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<Edge>,
    pub meta: GraphMeta,
}

#[derive(Serialize)]
pub struct GraphMeta {
    pub total_nodes: usize,
    pub total_edges: usize,
    /// Documentos que había en la selección antes de recortar.
    pub total_docs: usize,
    /// true si se mostraron solo los `max_nodes` más conectados.
    pub truncated: bool,
    pub bundles: Vec<String>,
}

struct DocRow {
    id: i64,
    slug: String,
    bundle: String,
    title: String,
    source_kind: String,
    tags: Vec<String>,
    links: Vec<String>,
}

/// Aristas por similitud: para cada documento, sus `SIM_NEIGHBORS`
/// vecinos más parecidos (coseno entre centroides CENTRADOS).
fn similarity_edges(
    conn: &Connection,
    bundle: Option<&str>,
    docs: &[DocRow],
) -> Result<Vec<Edge>, String> {
    let dim = crate::embed::EMBED_DIM;
    let index_of: FxHashMap<i64, usize> = docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();
    let mut sums: Vec<Vec<f32>> = vec![vec![0f32; dim]; docs.len()];
    let mut counts: Vec<u32> = vec![0; docs.len()];

    // Una sola consulta para todos los chunks (antes: un prepare + query
    // por documento). Solo se embebe lo que pertenece a un nodo visible.
    let mut stmt = conn
        .prepare(
            "SELECT c.doc_id, c.text FROM chunks c JOIN documents d ON d.id = c.doc_id \
             WHERE (?1 IS NULL OR d.bundle = ?1)",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![bundle], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })
        .map_err(|e| e.to_string())?;
    for row in rows.filter_map(Result::ok) {
        let (doc_id, text) = row;
        if let Some(&i) = index_of.get(&doc_id) {
            for (k, v) in crate::embed::embed(&text).into_iter().enumerate() {
                sums[i][k] += v;
            }
            counts[i] += 1;
        }
    }

    let valid: Vec<usize> = (0..docs.len()).filter(|&i| counts[i] > 0).collect();
    if valid.len() < 2 {
        return Ok(vec![]);
    }

    // Centroide por documento -> restar la media global -> normalizar.
    let mut mean = vec![0f32; dim];
    for &i in &valid {
        let n = counts[i] as f32;
        for k in 0..dim {
            sums[i][k] /= n;
            mean[k] += sums[i][k];
        }
    }
    for m in mean.iter_mut() {
        *m /= valid.len() as f32;
    }
    for &i in &valid {
        for k in 0..dim {
            sums[i][k] -= mean[k];
        }
        let norm = sums[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in sums[i].iter_mut() {
                *x /= norm;
            }
        }
    }

    let mut out = Vec::new();
    for &i in &valid {
        let mut sims: Vec<(f32, usize)> = valid
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| {
                let s: f32 = sums[i].iter().zip(sums[j].iter()).map(|(a, b)| a * b).sum();
                (s, j)
            })
            .collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(s, j) in sims.iter().take(SIM_NEIGHBORS) {
            if s < SIM_EDGE_MIN {
                break;
            }
            out.push(Edge {
                source: docs[i].id,
                target: docs[j].id,
                weight: s as f64,
                kind: "sim",
                priority: 1,
            });
        }
    }
    Ok(out)
}

pub fn build_okf_graph(
    conn: &Connection,
    bundle: Option<&str>,
    opts: GraphOpts,
) -> Result<OkfGraph, String> {
    let max_nodes = opts.max_nodes.clamp(50, 1500);

    let total: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM documents WHERE (?1 IS NULL OR bundle = ?1)",
            params![bundle],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    if total as usize > MAX_DOCS_FOR_PAIRWISE_SIM {
        let bundle_txt = bundle
            .map(|b| format!("El bundle '{b}'"))
            .unwrap_or_else(|| "La selección actual".to_string());
        return Err(format!(
            "{bundle_txt} tiene {total} documentos — demasiados para analizar (tope: {MAX_DOCS_FOR_PAIRWISE_SIM}). Elegí un bundle más chico en el filtro de arriba."
        ));
    }

    let sql = "SELECT id, slug, bundle, title, source_kind, tags, links FROM documents WHERE (?1 IS NULL OR bundle = ?1)";
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mut docs: Vec<DocRow> = stmt
        .query_map(params![bundle], |r| {
            let tags_json: String = r.get(5)?;
            let links_json: String = r.get(6)?;
            Ok(DocRow {
                id: r.get(0)?,
                slug: r.get(1)?,
                bundle: r.get(2)?,
                title: r.get(3)?,
                source_kind: r.get(4)?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                links: serde_json::from_str(&links_json).unwrap_or_default(),
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();

    if docs.is_empty() {
        return Ok(OkfGraph {
            nodes: vec![],
            edges: vec![],
            meta: GraphMeta {
                total_nodes: 0,
                total_edges: 0,
                total_docs: 0,
                truncated: false,
                bundles: vec![],
            },
        });
    }
    let total_docs = docs.len();

    // Índice de color por bundle, calculado ANTES de recortar para que
    // los colores no cambien según qué documentos queden.
    let mut bundles: Vec<String> = docs.iter().map(|d| d.bundle.clone()).collect();
    bundles.sort();
    bundles.dedup();
    let bundle_idx_of: FxHashMap<String, usize> = bundles
        .iter()
        .enumerate()
        .map(|(i, b)| (b.clone(), i))
        .collect();

    let mut cand: Vec<Edge> = vec![];

    // ── 1) Aristas por link explícita — O(n) ──
    {
        let slug_to_id: FxHashMap<&str, i64> =
            docs.iter().map(|d| (d.slug.as_str(), d.id)).collect();
        for d in &docs {
            for link in &d.links {
                let target_slug = link.rsplit('/').next().unwrap_or(link);
                if let Some(&tid) = slug_to_id.get(target_slug) {
                    if tid != d.id {
                        cand.push(Edge {
                            source: d.id,
                            target: tid,
                            weight: LINK_EDGE_WEIGHT,
                            kind: "link",
                            priority: 3,
                        });
                    }
                }
            }
        }
    }

    // ── 2) Aristas por tag compartido — UNA arista por par ──
    let pair_tags: FxHashMap<(i64, i64), (u32, f64)> = {
        let mut by_tag: FxHashMap<&str, Vec<i64>> = FxHashMap::default();
        for d in &docs {
            for t in &d.tags {
                by_tag.entry(t.as_str()).or_default().push(d.id);
            }
        }
        let mut pairs: FxHashMap<(i64, i64), (u32, f64)> = FxHashMap::default();
        for ids in by_tag.values_mut() {
            ids.sort_unstable();
            ids.dedup();
            if ids.len() < 2 || ids.len() > TAG_MAX_DOCS {
                continue;
            }
            // Un tag más raro (menos documentos) es una señal más fuerte.
            let rarity = 1.0 / ids.len() as f64;
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let e = pairs.entry((ids[i], ids[j])).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += rarity;
                }
            }
        }
        pairs
    };
    for ((a, b), (shared, rarity)) in pair_tags {
        cand.push(Edge {
            source: a,
            target: b,
            weight: TAG_EDGE_WEIGHT + 0.1 * (shared as f64 - 1.0) + 0.2 * rarity.min(1.0),
            kind: "tag",
            priority: 2,
        });
    }

    // ── 3) Recorte a los `max_nodes` documentos más conectados ──
    let truncated = docs.len() > max_nodes;
    if truncated {
        let mut deg: FxHashMap<i64, usize> = FxHashMap::default();
        for e in &cand {
            *deg.entry(e.source).or_insert(0) += 1;
            *deg.entry(e.target).or_insert(0) += 1;
        }
        docs.sort_by(|a, b| {
            let da = deg.get(&a.id).copied().unwrap_or(0);
            let db = deg.get(&b.id).copied().unwrap_or(0);
            db.cmp(&da).then(a.id.cmp(&b.id))
        });
        docs.truncate(max_nodes);
        let keep: FxHashSet<i64> = docs.iter().map(|d| d.id).collect();
        cand.retain(|e| keep.contains(&e.source) && keep.contains(&e.target));
    }

    // ── 4) Similitud por embeddings (opcional, solo sobre los nodos visibles) ──
    if opts.sim {
        cand.extend(similarity_edges(conn, bundle, &docs)?);
    }

    // ── 5) Selección global con tope simétrico por nodo ──
    // Orden: prioridad desc (link > tag > sim), luego peso desc.
    cand.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then(b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut deg: FxHashMap<i64, usize> = FxHashMap::default();
    let mut seen: FxHashSet<(i64, i64)> = FxHashSet::default();
    let mut edges: Vec<Edge> = vec![];
    for e in cand {
        let key = if e.source < e.target {
            (e.source, e.target)
        } else {
            (e.target, e.source)
        };
        if seen.contains(&key) {
            continue;
        }
        if deg.get(&e.source).copied().unwrap_or(0) >= MAX_EDGES_PER_NODE
            || deg.get(&e.target).copied().unwrap_or(0) >= MAX_EDGES_PER_NODE
        {
            continue;
        }
        seen.insert(key);
        *deg.entry(e.source).or_insert(0) += 1;
        *deg.entry(e.target).or_insert(0) += 1;
        edges.push(Edge {
            weight: (e.weight * 10000.0).round() / 10000.0,
            ..e
        });
    }

    let chunk_counts: FxHashMap<i64, i64> = conn
        .prepare("SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id")
        .map_err(|e| e.to_string())?
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();

    let nodes = docs
        .into_iter()
        .map(|d| GraphNode {
            degree: *deg.get(&d.id).unwrap_or(&0),
            bundle_idx: *bundle_idx_of.get(d.bundle.as_str()).unwrap_or(&0),
            chunks: *chunk_counts.get(&d.id).unwrap_or(&0),
            id: d.id,
            slug: d.slug,
            bundle: d.bundle,
            title: d.title,
            source_kind: d.source_kind,
            tags: d.tags,
            links: d.links,
        })
        .collect::<Vec<_>>();

    let meta = GraphMeta {
        total_nodes: nodes.len(),
        total_edges: edges.len(),
        total_docs,
        truncated,
        bundles,
    };
    Ok(OkfGraph { nodes, edges, meta })
}

// ─── Mapa de similitud: PCA 2D vía power iteration ──────────────────────
const MAX_POINTS_FOR_MAP: usize = 6000;

fn power_iteration_component(data: &[f32], dim: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    let mut state = seed | 1;
    for x in v.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *x = ((state % 2000) as f32 / 1000.0) - 1.0;
    }
    normalize(&mut v);
    for _ in 0..iters {
        let y: Vec<f32> = data
            .chunks_exact(dim)
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let mut v_new = vec![0f32; dim];
        for (row, &yi) in data.chunks_exact(dim).zip(y.iter()) {
            for (k, x) in row.iter().enumerate() {
                v_new[k] += x * yi;
            }
        }
        normalize(&mut v_new);
        v = v_new;
    }
    v
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[derive(Serialize)]
pub struct SimPoint {
    pub id: i64,
    pub doc_slug: String,
    pub doc_title: String,
    pub bundle: String,
    pub bundle_idx: usize,
    pub text: String,
    pub x: f32,
    pub y: f32,
}

#[derive(Serialize)]
pub struct SimilarityMap {
    pub points: Vec<SimPoint>,
    pub query: Option<(f32, f32)>,
    pub query_text: Option<String>,
    pub retrieved_ids: Vec<i64>,
    pub sampled: bool,
    pub sample_size: usize,
    pub total_chunks: i64,
}

/// A diferencia de v1, esto ya no depende de un `VectorCache` en RAM —
/// samplea directo por SQL (módulo sobre el rowid, determinístico) y
/// pide solo lo que hace falta para la proyección. Con sqlite-vec de
/// todos modos NO se necesita este sample para la búsqueda en sí — es
/// solo para que el scatter plot sea legible (más de unos miles de
/// puntos es ilegible igual).
pub fn build_similarity_map(
    conn: &Connection,
    vec_index: &crate::vec_index::VecIndex,
    bundle: Option<&str>,
    query: &str,
) -> rusqlite::Result<SimilarityMap> {
    let total: i64 = if let Some(b) = bundle {
        conn.query_row(
            "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.bundle=?1",
            params![b],
            |r| r.get(0),
        )?
    } else {
        conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?
    };
    if total == 0 {
        return Ok(SimilarityMap {
            points: vec![],
            query: None,
            query_text: None,
            retrieved_ids: vec![],
            sampled: false,
            sample_size: 0,
            total_chunks: 0,
        });
    }
    let stride = ((total as f64) / (MAX_POINTS_FOR_MAP as f64))
        .ceil()
        .max(1.0) as i64;

    struct Row {
        id: i64,
        text: String,
        slug: String,
        title: String,
        bundle: String,
    }

    // El filtro por módulo (`id % stride = 0`) obliga a un escaneo
    // completo — ningún índice ayuda con esa condición. Sin filtro de
    // bundle, los ids son (casi) contiguos 1..max_id, así que en vez de
    // escanear se generan posiciones parejas directas y se piden por su
    // primary key (O(log n) cada una) — con millones de chunks, la
    // diferencia es la que hacía que esto se quedara pegado/tumbara la
    // app. Con un bundle puntual filtrado, el índice de `documents.bundle`
    // ya acota bastante el escaneo, así que se mantiene el enfoque
    // anterior ahí.
    let rows: Vec<Row> = if bundle.is_none() {
        let max_id: i64 =
            conn.query_row("SELECT COALESCE(MAX(id),0) FROM chunks", [], |r| r.get(0))?;
        let min_id: i64 =
            conn.query_row("SELECT COALESCE(MIN(id),0) FROM chunks", [], |r| r.get(0))?;
        let mut sampled: Vec<Row> = Vec::new();
        if max_id > 0 {
            let sample_n = (MAX_POINTS_FOR_MAP as i64).min(total).max(1);
            let span = (max_id - min_id + 1) as f64;
            let step = (span / sample_n as f64).max(1.0);
            // Busca el primer fragmento existente desde cada posición de
            // muestreo. Mantiene búsquedas por PK (O(log n)) y evita que
            // huecos creados por borrados o rebuild dejen el mapa vacío.
            let mut stmt = conn.prepare(
                "SELECT c.id, c.text, d.slug, d.title, d.bundle FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id >= ?1 ORDER BY c.id LIMIT 1",
            )?;
            let mut last_id = -1i64;
            for i in 0..sample_n {
                let target_id = min_id + (i as f64 * step) as i64;
                if let Ok(row) = stmt.query_row(params![target_id], |r| {
                    Ok(Row {
                        id: r.get(0)?,
                        text: r.get(1)?,
                        slug: r.get(2)?,
                        title: r.get(3)?,
                        bundle: r.get(4)?,
                    })
                }) {
                    if row.id != last_id {
                        last_id = row.id;
                        sampled.push(row);
                    }
                }
            }
        }
        sampled
    } else {
        let sql = "SELECT c.id, c.text, d.slug, d.title, d.bundle FROM chunks c JOIN documents d ON d.id=c.doc_id
                    WHERE d.bundle=?1 AND (c.id % ?2) = 0";
        let mut stmt = conn.prepare(sql)?;
        let x = stmt
            .query_map(params![bundle.unwrap(), stride], |r| {
                Ok(Row {
                    id: r.get(0)?,
                    text: r.get(1)?,
                    slug: r.get(2)?,
                    title: r.get(3)?,
                    bundle: r.get(4)?,
                })
            })?
            .filter_map(Result::ok)
            .collect();
        x
    };

    if rows.is_empty() {
        return Ok(SimilarityMap {
            points: vec![],
            query: None,
            query_text: None,
            retrieved_ids: vec![],
            sampled: false,
            sample_size: 0,
            total_chunks: total,
        });
    }
    let dim = crate::vec_index::DIMENSIONS;
    let n = rows.len();
    let mut data: Vec<f32> = Vec::with_capacity(n * dim);
    for r in &rows {
        data.extend(crate::embed::embed(&r.text));
    }

    let query = query.trim();
    let mut query_idx = None;
    if !query.is_empty() {
        data.extend(crate::embed::embed(query));
        query_idx = Some(n);
    }
    let total_rows = data.len() / dim;

    let mut mean = vec![0f32; dim];
    for row in data.chunks_exact(dim) {
        for (m, x) in mean.iter_mut().zip(row.iter()) {
            *m += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= total_rows as f32;
    }
    for row in data.chunks_exact_mut(dim) {
        for (x, m) in row.iter_mut().zip(mean.iter()) {
            *x -= m;
        }
    }

    let comp1 = power_iteration_component(&data, dim, 12, 0x2545F4914F6CDD1D);
    let mut deflated = data.clone();
    for row in deflated.chunks_exact_mut(dim) {
        let proj: f32 = row.iter().zip(comp1.iter()).map(|(a, b)| a * b).sum();
        for (x, c) in row.iter_mut().zip(comp1.iter()) {
            *x -= proj * c;
        }
    }
    let comp2 = power_iteration_component(&deflated, dim, 12, 0x9E3779B97F4A7C15);

    let project = |row: &[f32]| -> (f32, f32) {
        let a: f32 = row.iter().zip(comp1.iter()).map(|(x, c)| x * c).sum();
        let b: f32 = row.iter().zip(comp2.iter()).map(|(x, c)| x * c).sum();
        (a, b)
    };

    let mut all_bundles: Vec<String> = {
        let mut stmt = conn.prepare("SELECT DISTINCT bundle FROM documents ORDER BY bundle")?;
        let bundles = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .filter_map(Result::ok)
            .collect();
        bundles
    };
    all_bundles.sort();
    all_bundles.dedup();
    let bundle_idx_of: FxHashMap<&str, usize> = all_bundles
        .iter()
        .enumerate()
        .map(|(i, b)| (b.as_str(), i))
        .collect();

    let mut points = vec![];
    for (i, row) in data.chunks_exact(dim).enumerate().take(n) {
        let (x, y) = project(row);
        let r = &rows[i];
        let snippet: String = r
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(200)
            .collect();
        points.push(SimPoint {
            id: r.id,
            doc_slug: r.slug.clone(),
            doc_title: r.title.clone(),
            bundle: r.bundle.clone(),
            bundle_idx: *bundle_idx_of.get(r.bundle.as_str()).unwrap_or(&0),
            text: snippet,
            x,
            y,
        });
    }

    let mut query_point = None;
    let mut retrieved_ids = vec![];
    if let Some(qi) = query_idx {
        let row = &data[qi * dim..(qi + 1) * dim];
        query_point = Some(project(row));
        retrieved_ids = crate::search::hybrid_search(conn, vec_index, query, bundle, 6)
            .into_iter()
            .map(|r| r.id)
            .collect();
    }

    Ok(SimilarityMap {
        points,
        query: query_point,
        query_text: if query.is_empty() {
            None
        } else {
            Some(query.to_string())
        },
        retrieved_ids,
        sampled: (total as usize) > n,
        sample_size: n,
        total_chunks: total,
    })
}