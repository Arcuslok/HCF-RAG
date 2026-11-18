//! reindex.rs — pipeline paralelo: varios hilos preparan documentos
//! (leer + parsear + chunkear + embeddear, todo puro, sin tocar
//! ninguna base) y UN hilo escritor hace los INSERT a SQLite +
//! vec_index.add(), agrupando cientos de documentos por transacción.

use rusqlite::Connection;
use rustc_hash::FxHashSet;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use crate::db::{self, PreparedInsert};
use crate::embed::{chunk_text, embed};
use crate::frontmatter::parse_okf;
use crate::vec_index::VecIndex;

#[derive(Serialize, Default)]
pub struct ReindexStats {
    pub scanned: i64,
    pub added: i64,
    pub updated: i64,
    pub removed: i64,
    pub unchanged: i64,
    pub errors: i64,
}

#[derive(Serialize, Default)]
pub struct RebuildStats {
    pub bundles: i64,
    pub documents: i64,
    pub chunks: i64,
    pub errors: i64,
}

const COMMIT_BATCH: usize = 500;
/// Cuántos errores reales (con el mensaje, no solo el conteo) se
/// loguean por corrida — con millones de archivos, loguear cada error
/// inundaría el log sin aportar nada después de los primeros; si el
/// problema es sistemático (un bug, no un archivo puntual raro), con
/// unos pocos alcanza para diagnosticarlo.
const MAX_LOGGED_ERRORS: usize = 20;
/// Cada cuántos lotes se persiste el índice de vectores a disco — no
/// conviene hacerlo por lote (usearch reescribe el archivo entero cada
/// save(), carísimo con millones de vectores); esto es un compromiso
/// entre "poco perdido si se cae el proceso" y "no perder tiempo
/// serializando todo el rato".
const VEC_SAVE_EVERY_BATCHES: usize = 20;

fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .saturating_sub(1)
        .max(1)
}

/// Trabajo puro (sin tocar ninguna base) que puede correr en cualquier
/// hilo: leer el .md, parsear el frontmatter, chunkear y calcular
/// embeddings (floats crudos, ya no hace falta serializarlos a bytes).
fn prepare_document(path: &Path, bundle: &str) -> Result<PreparedInsert, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let parsed = parse_okf(&text);
    let slug = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let title = if parsed.title.is_empty() {
        slug.clone()
    } else {
        parsed.title
    };
    let chunks: Vec<(String, Vec<f32>)> = chunk_text(&parsed.body)
        .into_iter()
        .map(|ch| {
            let emb = embed(&ch);
            (ch, emb)
        })
        .collect();
    Ok(PreparedInsert {
        slug,
        bundle: bundle.to_string(),
        title,
        source_kind: parsed.source_kind,
        intent_kind: parsed.intent_kind,
        tags: parsed.tags,
        links: parsed.links,
        content_hash: parsed.hash,
        okf_path: path.to_path_buf(),
        chunks,
    })
}

/// Baja `synchronous` a OFF durante el bulk load y lo restaura al salir
/// del scope — seguro acá porque todo esto es *derived data*.
struct FastBulkGuard<'a>(&'a Connection);
impl<'a> FastBulkGuard<'a> {
    fn new(conn: &'a Connection) -> Self {
        let _ = conn.execute_batch("PRAGMA synchronous=OFF;");
        FastBulkGuard(conn)
    }
}
impl<'a> Drop for FastBulkGuard<'a> {
    fn drop(&mut self) {
        let _ = self.0.execute_batch("PRAGMA synchronous=NORMAL;");
    }
}

pub fn rebuild_index_parallel(
    conn: &Connection,
    vec_index: &mut VecIndex,
    bundle_dir: &Path,
    progress: &AtomicI64,
    total: &AtomicI64,
) -> Result<RebuildStats, String> {
    db::wipe_index(conn, vec_index)?;
    let _fast = FastBulkGuard::new(conn);
    let mut stats = RebuildStats::default();

    let bundle_dirs = db::list_bundle_dirs(bundle_dir);

    // ── Contar archivos por bundle para reservar capacidad en cada índice ──
    let mut bundle_files: Vec<(PathBuf, Vec<PathBuf>, String)> = Vec::new();
    let mut grand_total = 0i64;
    for dir in &bundle_dirs {
        let bundle = dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        let files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
            .collect();
        if !files.is_empty() {
            let n_files = files.len();
            // Reservar capacidad estimada (~1.3 chunks por archivo en promedio)
            let _ = vec_index.reserve(&bundle, (n_files as f64 * 1.3) as usize);
            grand_total += n_files as i64;
            bundle_files.push((dir.clone(), files, bundle));
        }
    }
    total.store(grand_total, Ordering::Relaxed);

    let mut batches_since_save = 0usize;
    let mut logged_errors = 0usize;

    for (_dir, files, bundle) in bundle_files {
        stats.bundles += 1;
        if files.is_empty() {
            continue;
        }

        let files = Arc::new(files);
        let next_idx = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::sync_channel::<Result<PreparedInsert, String>>(COMMIT_BATCH * 2);
        let n_workers = worker_count();

        let handles: Vec<_> = (0..n_workers)
            .map(|_| {
                let files = Arc::clone(&files);
                let next_idx = Arc::clone(&next_idx);
                let tx = tx.clone();
                let bundle = bundle.clone();
                std::thread::spawn(move || loop {
                    let i = next_idx.fetch_add(1, Ordering::Relaxed);
                    if i >= files.len() {
                        break;
                    }
                    let _ = tx.send(prepare_document(&files[i], &bundle));
                })
            })
            .collect();
        drop(tx);

        let mut in_txn = false;
        let mut since_commit = 0usize;
        for result in rx {
            progress.fetch_add(1, Ordering::Relaxed);
            if !in_txn {
                conn.execute_batch("BEGIN IMMEDIATE")
                    .map_err(|e| e.to_string())?;
                in_txn = true;
            }
            match result {
                Ok(prepared) => match db::insert_prepared(conn, vec_index, &prepared) {
                    Ok((_id, n)) => {
                        stats.documents += 1;
                        stats.chunks += n as i64;
                    }
                    Err(e) => {
                        stats.errors += 1;
                        if logged_errors < MAX_LOGGED_ERRORS {
                            crate::plugin_log(&format!(
                                "ERROR rebuild insert_prepared '{}': {e}",
                                prepared.slug
                            ));
                            logged_errors += 1;
                        }
                    }
                },
                Err(e) => {
                    stats.errors += 1;
                    if logged_errors < MAX_LOGGED_ERRORS {
                        crate::plugin_log(&format!("ERROR rebuild prepare_document: {e}"));
                        logged_errors += 1;
                    }
                }
            }
            since_commit += 1;
            if since_commit >= COMMIT_BATCH {
                conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;
                in_txn = false;
                since_commit = 0;
                batches_since_save += 1;
                if batches_since_save >= VEC_SAVE_EVERY_BATCHES {
                    let _ = vec_index.save();
                    batches_since_save = 0;
                }
            }
        }
        if in_txn {
            conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;
        }
        for h in handles {
            let _ = h.join();
        }
    }
    let _ = vec_index.save();
    Ok(stats)
}

pub fn sync_from_bundle_parallel(
    conn: &Connection,
    vec_index: &mut VecIndex,
    bundle_dir: &Path,
    bundle: &str,
    progress: &AtomicI64,
    total: &AtomicI64,
) -> Result<ReindexStats, String> {
    if !crate::frontmatter::valid_bundle_name(bundle) {
        return Err("Nombre de bundle inválido".to_string());
    }
    let bundle_path = bundle_dir.join(bundle);
    if !bundle_path.exists() {
        return Err(format!(
            "El bundle '{bundle}' no existe en {}",
            bundle_path.display()
        ));
    }
    let _fast = FastBulkGuard::new(conn);
    let mut stats = ReindexStats::default();

    let files: Vec<PathBuf> = std::fs::read_dir(&bundle_path)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
        .collect();

    // Reservar capacidad estimada para este bundle
    let _ = vec_index.reserve(bundle, (files.len() as f64 * 1.3) as usize);
    total.store(files.len() as i64, Ordering::Relaxed);

    let db_hashes = Arc::new(db::document_hash_map(conn, bundle).map_err(|e| e.to_string())?);
    let mut disk_slugs: FxHashSet<String> =
        FxHashSet::with_capacity_and_hasher(files.len(), Default::default());

    let files = Arc::new(files);
    let next_idx = Arc::new(AtomicUsize::new(0));
    let (tx, rx) =
        mpsc::sync_channel::<(String, Result<Option<PreparedInsert>, String>)>(COMMIT_BATCH * 2);
    let n_workers = worker_count();
    let bundle_owned = bundle.to_string();

    let handles: Vec<_> = (0..n_workers)
        .map(|_| {
            let files = Arc::clone(&files);
            let next_idx = Arc::clone(&next_idx);
            let tx = tx.clone();
            let bundle = bundle_owned.clone();
            let db_hashes = Arc::clone(&db_hashes);
            std::thread::spawn(move || loop {
                let i = next_idx.fetch_add(1, Ordering::Relaxed);
                if i >= files.len() {
                    break;
                }
                let path = &files[i];
                let slug = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                match std::fs::read_to_string(path) {
                    Ok(text) => {
                        let disk_hash = parse_okf(&text).hash;
                        let unchanged = db_hashes
                            .get(&slug)
                            .map(|h| h == &disk_hash)
                            .unwrap_or(false);
                        if unchanged {
                            let _ = tx.send((slug, Ok(None)));
                        } else {
                            let _ = tx.send((slug, prepare_document(path, &bundle).map(Some)));
                        }
                    }
                    Err(e) => {
                        let _ = tx.send((slug, Err(e.to_string())));
                    }
                }
            })
        })
        .collect();
    drop(tx);

    let mut in_txn = false;
    let mut since_commit = 0usize;
    let mut batches_since_save = 0usize;
    let mut logged_errors = 0usize;
    for (slug, result) in rx {
        progress.fetch_add(1, Ordering::Relaxed);
        stats.scanned += 1;
        disk_slugs.insert(slug.clone());
        if !in_txn {
            conn.execute_batch("BEGIN IMMEDIATE")
                .map_err(|e| e.to_string())?;
            in_txn = true;
        }
        match result {
            Ok(None) => stats.unchanged += 1,
            Ok(Some(prepared)) => {
                let is_update = db_hashes.contains_key(&slug);
                match db::insert_prepared(conn, vec_index, &prepared) {
                    Ok(_) => {
                        if is_update {
                            stats.updated += 1
                        } else {
                            stats.added += 1
                        }
                    }
                    Err(e) => {
                        stats.errors += 1;
                        if logged_errors < MAX_LOGGED_ERRORS {
                            crate::plugin_log(&format!(
                                "ERROR reindex insert_prepared '{slug}': {e}"
                            ));
                            logged_errors += 1;
                        }
                    }
                }
            }
            Err(e) => {
                stats.errors += 1;
                if logged_errors < MAX_LOGGED_ERRORS {
                    crate::plugin_log(&format!("ERROR reindex '{slug}': {e}"));
                    logged_errors += 1;
                }
            }
        }
        since_commit += 1;
        if since_commit >= COMMIT_BATCH {
            conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;
            in_txn = false;
            since_commit = 0;
            batches_since_save += 1;
            if batches_since_save >= VEC_SAVE_EVERY_BATCHES {
                let _ = vec_index.save();
                batches_since_save = 0;
            }
        }
    }
    if in_txn {
        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;
    }
    for h in handles {
        let _ = h.join();
    }

    for slug in db_hashes.keys() {
        if !disk_slugs.contains(slug) {
            let _ = db::delete_document(conn, vec_index, slug, bundle);
            stats.removed += 1;
        }
    }
    let _ = vec_index.save();
    Ok(stats)
}
