//! ame_rag_plugin — RAG local como DLL nativa, con bundles, links
//! explícitas, reindex/rebuild desde OKF, y un índice de vectores
//! separado (usearch, HNSW) para la búsqueda semántica.
//!
//! Reglas de diseño que quedaron de las vueltas anteriores (ver README
//! para la historia completa):
//! - Ninguna llamada FFI bloquea más que unos milisegundos: las DLL no
//!   tienen timeout de protección del lado del host — por eso
//!   reindex/rebuild corren en background (ver JOB_*) en vez de bloquear
//!   la llamada.
//! - Los vectores NO viven en el `.sqlite3` (ver vec_index.rs) — es la
//!   fuente de la mayor parte del tamaño en disco cuando vivían ahí
//!   (BLOB f32 sin cuantizar, duplicado en un índice aparte). Ahora
//!   viven en `vectors.tvim` (nombre heredado de una vuelta anterior por
//!   turbovec — el archivo es 100% usearch/HNSW, cuantizado a i8; el
//!   nombre es cosmético, no hace falta migrarlo), con la key =
//!   `chunks.id` de SQLite.
//! - El `.md` en `Brain/Memory/Knowledge/bundle/<bundle>/<slug>.md` es la
//!   fuente de verdad; `.sqlite3` + `.tvim` son derivados y
//!   reconstruibles (`rag_rebuild_index`).

mod db;
pub mod embed;
mod frontmatter;
mod graph;
mod qa;
mod reindex;
mod search;
mod vec_index;

use rusqlite::Connection;
use serde_json::{json, Value};
use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;
use vec_index::VecIndex;

const LOADING_MSG: &str =
    "[i] Todavía estoy iniciando la base de conocimiento local. Probá de nuevo en un momento.";

// ═══════════════════════════════════════════════════════════════════════
// Logging propio del plugin — mismo puente que usa el plugin "Logs"
// (ver su lib.rs: el host llama a ame_plugin_set_logger con un callback,
// nosotros lo guardamos y lo llamamos con nuestro propio nombre — el host
// escribe a Brain/Logs/<fecha>_plugin_rag.log a partir de eso).
// ═══════════════════════════════════════════════════════════════════════
type HostLogFn = extern "C" fn(*const c_char, *const c_char);
static LOGGER: Mutex<Option<HostLogFn>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn ame_plugin_set_logger(logger: HostLogFn) {
    *LOGGER.lock().unwrap() = Some(logger);
}

pub(crate) fn plugin_log(msg: &str) {
    let logger = match *LOGGER.lock().unwrap() {
        Some(l) => l,
        None => return,
    };
    if let (Ok(name), Ok(line)) = (CString::new("RAG"), CString::new(msg)) {
        logger(name.as_ptr(), line.as_ptr());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Job en background (reindex/rebuild) — reindex/rebuild pueden tardar
// horas con bundles grandes (millones de .md). Antes corrían sincrónico
// sobre el hilo que atiende la llamada FFI, y como las DLL no tienen
// timeout de protección del lado del host, eso congelaba TODA la app
// ("No responde"). Ahora se lanzan en su propio hilo: el comando vuelve
// al toque con un ack, y el progreso se consulta con `rag_job_status`.
// Solo uno corre a la vez (reindex y rebuild comparten este estado)
// porque ambos tocan la misma tabla `documents` — correrlos en paralelo
// entre sí sería una carrera de escrituras sobre el mismo índice.
// ═══════════════════════════════════════════════════════════════════════
static JOB_RUNNING: Mutex<Option<String>> = Mutex::new(None);
static JOB_PROGRESS: AtomicI64 = AtomicI64::new(0);
static JOB_TOTAL: AtomicI64 = AtomicI64::new(0);
static JOB_STARTED: Mutex<Option<Instant>> = Mutex::new(None);
static JOB_RESULT: Mutex<Option<String>> = Mutex::new(None);

/// Intenta tomar el slot de "job corriendo". Devuelve Err con un mensaje
/// listo para el usuario si ya hay uno en curso.
fn try_start_job(label: &str) -> Result<(), String> {
    let mut running = JOB_RUNNING.lock().unwrap();
    if let Some(current) = running.as_ref() {
        return Err(format!("[i] Ya hay un trabajo en curso ({current}) — esperá a que termine (rag_job_status para ver el progreso)."));
    }
    *running = Some(label.to_string());
    JOB_PROGRESS.store(0, Ordering::Relaxed);
    JOB_TOTAL.store(0, Ordering::Relaxed);
    *JOB_STARTED.lock().unwrap() = Some(Instant::now());
    *JOB_RESULT.lock().unwrap() = None;
    Ok(())
}

fn finish_job(msg: String) {
    *JOB_RESULT.lock().unwrap() = Some(msg);
    *JOB_RUNNING.lock().unwrap() = None;
}

fn cmd_wal_checkpoint() -> String {
    with_db(|conn| {
        match conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
            r.get::<_, i64>(0)
        }) {
            Ok(_) => {
                let msg =
                    "[OK] Checkpoint hecho — el .sqlite3-wal debería haberse achicado.".to_string();
                plugin_log(&msg);
                msg
            }
            Err(e) => {
                plugin_log(&format!("ERROR wal_checkpoint: {e}"));
                format!("[x] {e}")
            }
        }
    })
    .unwrap_or_else(|e| e)
}

fn cmd_vacuum() -> String {
    if let Err(e) = try_start_job("vacuum del .sqlite3") {
        return e;
    }
    plugin_log("vacuum iniciado en background");
    std::thread::spawn(|| {
        let outcome = with_db(|conn| conn.execute_batch("VACUUM"));
        let msg = match outcome {
            Ok(Ok(())) => {
                let m = "[OK] VACUUM terminado — el archivo debería haberse achicado al tamaño real de los datos.".to_string();
                plugin_log(&m);
                m
            }
            Ok(Err(e)) => {
                plugin_log(&format!("ERROR vacuum: {e}"));
                format!("[x] {e}")
            }
            Err(e) => {
                plugin_log(&format!("ERROR vacuum (host no listo): {e}"));
                format!("[x] {e}")
            }
        };
        finish_job(msg);
    });
    "[OK] VACUUM iniciado en background — puede tardar bastante en un archivo grande. Necesita ~el mismo espacio libre en disco que el tamaño actual del .sqlite3 mientras corre. Consultá el progreso con rag_job_status (sin ETA, VACUUM no reporta avance incremental).".to_string()
}

/// Tasa y ETA reales calculadas sobre el progreso medido — nada de
/// estimaciones a ojo: se recalculan en cada consulta a partir de
/// cuánto se hizo en cuánto tiempo, igual que hacía normalize_and_tag.py
/// por consola.
fn cmd_job_status() -> String {
    let running = JOB_RUNNING.lock().unwrap().clone();
    let processed = JOB_PROGRESS.load(Ordering::Relaxed);
    let total = JOB_TOTAL.load(Ordering::Relaxed);
    let last_result = JOB_RESULT.lock().unwrap().clone();
    let elapsed_secs = JOB_STARTED
        .lock()
        .unwrap()
        .map(|t| t.elapsed().as_secs_f64());

    let (rate_per_sec, eta_seconds) = match elapsed_secs {
        Some(e) if e > 0.5 && processed > 0 => {
            let rate = processed as f64 / e;
            let eta = if total > 0 {
                Some(((total - processed).max(0) as f64 / rate).round() as i64)
            } else {
                None
            };
            (Some((rate * 10.0).round() / 10.0), eta)
        }
        _ => (None, None),
    };

    json!({
        "running": running,
        "documents_processed": processed,
        "documents_total": total,
        "elapsed_seconds": elapsed_secs.map(|e| e.round() as i64),
        "rate_per_second": rate_per_sec,
        "eta_seconds": eta_seconds,
        "last_result": last_result,
    })
    .to_string()
}

struct Paths {
    bundle_dir: PathBuf,
    knowledge_dir: PathBuf,
}

// ═══════════════════════════════════════════════════════════════════════
// Locks separados en vez de un único Mutex<State>.
//
// Antes: conn + vec_index + paths vivían juntos detrás de un solo
// Mutex, tomado entero por CUALQUIER comando (incluso los que solo
// necesitaban una de las tres cosas). El precalentamiento del índice
// de vectores al boot (ver más abajo) podía tardar bastante con
// millones de vectores fríos (mmap sin páginas todavía en RAM — ver
// el hallazgo sobre usearch#514 documentado abajo) — y mientras
// duraba, comandos que solo necesitaban SQLite (rag_stats,
// rag_db_stats: el header "N documentos / N fragmentos" que pinta la
// UI apenas se abre el panel) se quedaban esperando ese mismo lock
// sin ninguna razón real. Eso era el freeze al abrir A.M.E./entrar a
// la pestaña RAG.
//
// Ahora: DB (Mutex<Connection>, sigue exclusivo — una Connection de
// SQLite no es segura para llamadas concurrentes) y VEC
// (RwLock<VecIndex>, admite MUCHOS lectores a la vez — varias
// búsquedas o un precalentamiento no se bloquean entre sí, solo
// add_batch/remove/save necesitan el lock de escritura). PATHS es
// inmutable una vez seteado, no necesita lock.
static PATHS: OnceLock<Paths> = OnceLock::new();
static DB: OnceLock<Mutex<Connection>> = OnceLock::new();
static VEC: OnceLock<RwLock<VecIndex>> = OnceLock::new();

enum InitState {
    NotStarted,
    Loading,
    Ready,
    Failed(String),
}

static INIT_STATE: Mutex<InitState> = Mutex::new(InitState::NotStarted);

fn paths() -> &'static Paths {
    PATHS.get().expect(
        "PATHS se setea junto con DB/VEC en start_background_init, antes de InitState::Ready",
    )
}

/// Expuesto para search.rs: el .json de marcadores de intención vive
/// junto al resto de los datos del RAG (Knowledge/), no embebido en el
/// binario, para que se pueda editar/ampliar sin recompilar.
pub(crate) fn knowledge_dir() -> &'static Path {
    &paths().knowledge_dir
}

fn resolve_knowledge_dir() -> PathBuf {
    if let Ok(p) = std::env::var("AME_KNOWLEDGE_DIR") {
        return PathBuf::from(p);
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));
    let base = exe_dir.unwrap_or_else(|| PathBuf::from("."));
    base.join("Brain").join("Memory").join("Knowledge")
}

fn start_background_init() {
    {
        let mut guard = INIT_STATE.lock().unwrap();
        match *guard {
            InitState::NotStarted => *guard = InitState::Loading,
            _ => return,
        }
    }
    std::thread::spawn(|| {
        let result = (|| -> Result<(), String> {
            let knowledge_dir = resolve_knowledge_dir();
            let bundle_dir = knowledge_dir.join("bundle");
            std::fs::create_dir_all(&bundle_dir).map_err(|e| e.to_string())?;
            let db_path = knowledge_dir.join("index.sqlite3");
            let conn = db::open(&db_path).map_err(|e| e.to_string())?;

            // CAMBIO: ahora es un directorio (vectors/), no un archivo (vectors.tvim)
            let vec_dir = knowledge_dir.join("vectors");
            let vec_index = vec_index::VecIndex::open(&vec_dir).map_err(|e| e.to_string())?;

            let _ = PATHS.set(Paths {
                bundle_dir,
                knowledge_dir,
            });
            let _ = DB.set(Mutex::new(conn));
            let _ = VEC.set(RwLock::new(vec_index));
            Ok(())
        })();
        let mut init_ok = false;
        {
            let mut guard = INIT_STATE.lock().unwrap();
            *guard = match result {
                Ok(()) => {
                    let n_vectors = VEC.get().map(|v| v.read().unwrap().len()).unwrap_or(0);
                    plugin_log(&format!(
                        "inicializado OK — bundle_dir={} vectores={}",
                        paths().bundle_dir.display(),
                        n_vectors
                    ));
                    init_ok = true;
                    InitState::Ready
                }
                Err(e) => {
                    plugin_log(&format!("FALLÓ la inicialización: {e}"));
                    InitState::Failed(e)
                }
            };
        }
        if init_ok {
            let _ = with_vec_read(|vec| {
                let query_embedding = embed::embed("precalentamiento inicial de A.M.E.");
                vec.search(None, &query_embedding, 1)
            });
            plugin_log("precalentamiento del índice de vectores terminado");
        }
    });

    // Autosave cada 60s
    std::thread::spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
        let job_active = JOB_RUNNING.lock().unwrap().is_some();
        if let Some(vec_lock) = VEC.get() {
            let mut vec = vec_lock.write().unwrap();
            let result = if job_active {
                vec.save()
            } else {
                vec.release_ram()
            };
            if let Err(e) = result {
                plugin_log(&format!("ERROR autosave del índice de vectores: {e}"));
            }
        }
    });
}

// ─── cmd_db_stats actualizado para recorrer subdirectorios de vectores ───

fn dir_size(path: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_file() {
            total += meta.len();
        } else if meta.is_dir() {
            total += dir_size(&entry.path())?;
        }
    }
    Ok(total)
}

fn cmd_db_stats() -> String {
    with_db_and_vec_read(|conn, vec| {
        let page_count: i64 = conn.query_row("PRAGMA page_count", [], |r| r.get(0)).unwrap_or(0);
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |r| r.get(0)).unwrap_or(0);
        let freelist_count: i64 = conn.query_row("PRAGMA freelist_count", [], |r| r.get(0)).unwrap_or(0);
        let total_bytes = page_count * page_size;
        let reclaimable_bytes = freelist_count * page_size;
        let n_docs: i64 = conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)).unwrap_or(0);
        let n_chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)).unwrap_or(0);

        // Recorrer todos los subdirectorios de vectors/ para sumar tamaño
        let mut vec_bytes = 0u64;
        let vec_dir = paths().knowledge_dir.join("vectors");
        if let Ok(entries) = std::fs::read_dir(&vec_dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(size) = dir_size(&path) {
                        vec_bytes += size;
                    }
                }
            }
        }

        json!({
            "total_bytes": total_bytes,
            "reclaimable_bytes": reclaimable_bytes,
            "reclaimable_pct": if total_bytes > 0 { (reclaimable_bytes as f64 / total_bytes as f64 * 100.0 * 10.0).round() / 10.0 } else { 0.0 },
            "documents": n_docs,
            "chunks": n_chunks,
            "vectors_in_index": vec.len(),
            "vec_index_bytes": vec_bytes,
        })
        .to_string()
    })
    .unwrap_or_else(|e| e)
}

fn ensure_ready() -> Result<(), String> {
    start_background_init();
    let guard = INIT_STATE.lock().unwrap();
    match &*guard {
        InitState::Ready => Ok(()),
        InitState::Loading | InitState::NotStarted => Err(LOADING_MSG.to_string()),
        InitState::Failed(e) => Err(format!("[x] No se pudo inicializar el RAG: {e}")),
    }
}

/// Solo SQLite. Usado por comandos que no tocan el índice de vectores
/// (rag_stats, rag_db_stats — salvo el conteo de vectores, rag_list,
/// rag_list_bundles, etc.) — nunca esperan detrás de una búsqueda o
/// un precalentamiento en curso.
fn with_db<T>(f: impl FnOnce(&Connection) -> T) -> Result<T, String> {
    ensure_ready()?;
    let conn = DB.get().unwrap().lock().unwrap();
    Ok(f(&conn))
}

/// Solo índice de vectores, lectura. RwLock: varios lectores
/// concurrentes está bien (varias búsquedas a la vez, o una búsqueda
/// mientras corre el precalentamiento).
fn with_vec_read<T>(f: impl FnOnce(&VecIndex) -> T) -> Result<T, String> {
    ensure_ready()?;
    let vec = VEC.get().unwrap().read().unwrap();
    Ok(f(&vec))
}

/// Solo índice de vectores, escritura exclusiva (add/remove/save/
/// release_ram).
#[allow(dead_code)]
fn with_vec_write<T>(f: impl FnOnce(&mut VecIndex) -> T) -> Result<T, String> {
    ensure_ready()?;
    let mut vec = VEC.get().unwrap().write().unwrap();
    Ok(f(&mut vec))
}

/// Búsqueda híbrida y similares, que necesitan SQLite y vectores a la
/// vez. Orden de locks fijo en TODOS los helpers combinados (conn
/// primero, vec después) para no arriesgar un deadlock cruzado con
/// otro comando que los tomara al revés.
fn with_db_and_vec_read<T>(f: impl FnOnce(&Connection, &VecIndex) -> T) -> Result<T, String> {
    ensure_ready()?;
    let conn = DB.get().unwrap().lock().unwrap();
    let vec = VEC.get().unwrap().read().unwrap();
    Ok(f(&conn, &vec))
}

/// Igual, pero con escritura exclusiva sobre el índice de vectores
/// (ingest/delete/reindex/rebuild — todos tocan documents/chunks en
/// SQLite Y agregan/borran vectores).
fn with_db_and_vec_write<T>(f: impl FnOnce(&Connection, &mut VecIndex) -> T) -> Result<T, String> {
    ensure_ready()?;
    let conn = DB.get().unwrap().lock().unwrap();
    let mut vec = VEC.get().unwrap().write().unwrap();
    Ok(f(&conn, &mut vec))
}

/// Los comandos aceptan JSON (`{"query": "...", "bundle": "..."}`) O un
/// string plano (se toma como el campo principal, bundle=None) — así la
/// UI puede mandar JSON estructurado y `rag_search algo` a mano desde el
/// chat sigue funcionando.
fn parse_args(args: &str) -> Value {
    serde_json::from_str(args).unwrap_or_else(|_| json!({"_plain": args.trim()}))
}
fn field_or_plain<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    for k in keys {
        if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    v.get("_plain")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
}
fn str_list(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|i| i.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}
fn bundle_of(v: &Value) -> Result<String, String> {
    let bundle = v
        .get("bundle")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("default");
    if frontmatter::valid_bundle_name(bundle) {
        Ok(bundle.to_string())
    } else {
        Err("[x] Nombre de bundle inválido — solo letras, números, '-' y '_' (sin espacios ni símbolos).".into())
    }
}
fn bundle_filter_of(v: &Value) -> Option<String> {
    v.get("bundle")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from)
}
/// Clampea a uno de los 3 cubos conocidos ("info"/"herramienta"/"plan")
/// o "" (sin clasificar) — nunca deja pasar un valor arbitrario a
/// disco/DB, para que `intent_boost` en search.rs pueda comparar por
/// igualdad exacta sin tener que normalizar del otro lado.
fn intent_kind_of(v: &Value) -> String {
    match v.get("intent_kind").and_then(|x| x.as_str()) {
        Some("info") => "info".to_string(),
        Some("herramienta") => "herramienta".to_string(),
        Some("plan") => "plan".to_string(),
        _ => String::new(),
    }
}

// ─── Comandos ─────────────────────────────────────────────────────────
fn cmd_ingest_text(args: &str) -> String {
    let v = parse_args(args);
    let Some(text) = v
        .get("text")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
    else {
        return "[x] No hay texto para ingestar.".into();
    };
    let title = v
        .get("title")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("Documento sin título");
    let tags = str_list(&v, "tags");
    let links = str_list(&v, "links");
    let bundle = match bundle_of(&v) {
        Ok(bundle) => bundle,
        Err(e) => return e,
    };
    let intent_kind = intent_kind_of(&v);

    with_db_and_vec_write(|conn, vec| {
        match db::ingest(
            conn,
            vec,
            &paths().bundle_dir,
            &bundle,
            title,
            text,
            "text",
            &intent_kind,
            &tags,
            &links,
        ) {
            Ok((_id, slug, n)) => {
                let msg = format!(
                    "[OK] Ingestado '{title}' en bundle '{bundle}' — {n} fragmento(s), id={slug}."
                );
                plugin_log(&msg);
                msg
            }
            Err(e) => {
                plugin_log(&format!(
                    "ERROR ingest_text '{title}' bundle '{bundle}': {e}"
                ));
                format!("[x] {e}")
            }
        }
    })
    .unwrap_or_else(|e| e)
}

/// Extrae el texto de un PDF a partir de sus bytes crudos.
///
/// Usa `pdf-extract` (puro Rust, sin dependencias de sistema como
/// poppler/pdfium) — suficiente para PDFs con capa de texto real
/// (la inmensa mayoría de documentos generados digitalmente). PDFs
/// escaneados sin OCR no van a dar texto útil; eso queda fuera de
/// alcance del plugin nativo (necesitaría un motor de OCR aparte).
///
/// `pdf-extract` normalmente devuelve el texto con saltos de línea y
/// espaciado bastante ruidoso (uno por glifo posicionado, no por
/// palabra/párrafo real), así que se normaliza antes de mandarlo a
/// `db::ingest` — si no, el chunking de párrafos en frontmatter.rs
/// terminaría cortando de forma rarísima.
fn extract_pdf_text(bytes: &[u8]) -> Result<String, String> {
    let raw = pdf_extract::extract_text_from_mem(bytes).map_err(|e| e.to_string())?;
    Ok(normalize_pdf_text(&raw))
}

/// Colapsa espacios/tabs repetidos dentro de cada línea, recorta
/// bordes, y reduce 3+ líneas vacías seguidas (saltos de página /
/// artefactos de layout) a una sola línea en blanco — conserva los
/// párrafos reales (una línea en blanco) para que el chunking de
/// frontmatter.rs los siga reconociendo.
fn normalize_pdf_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut blank_run = 0u32;
    for line in raw.lines() {
        let collapsed: String = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                out.push('\n');
            }
        } else {
            blank_run = 0;
            out.push_str(&collapsed);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

fn cmd_ingest_file(args: &str) -> String {
    let v = parse_args(args);
    let filename = v
        .get("filename")
        .and_then(|x| x.as_str())
        .unwrap_or("archivo")
        .to_string();
    let Some(content_b64) = v
        .get("content_b64")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
    else {
        return "[x] No llegó contenido del archivo.".into();
    };
    let raw = match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, content_b64)
    {
        Ok(r) => r,
        Err(e) => return format!("[x] base64 inválido: {e}"),
    };
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();
    let text = if ext == "pdf" {
        match extract_pdf_text(&raw) {
            Ok(t) => t,
            Err(e) => {
                plugin_log(&format!("ERROR extract_pdf_text '{filename}': {e}"));
                return format!("[x] No se pudo extraer texto del PDF '{filename}': {e}");
            }
        }
    } else {
        String::from_utf8_lossy(&raw).to_string()
    };
    if text.trim().is_empty() {
        return format!(
            "[x] No se pudo extraer texto de '{filename}' (¿PDF escaneado/sin capa de texto?)."
        );
    }
    let title = v
        .get("title")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&filename)
        .to_string();
    let tags = str_list(&v, "tags");
    let links = str_list(&v, "links");
    let bundle = match bundle_of(&v) {
        Ok(bundle) => bundle,
        Err(e) => return e,
    };
    let intent_kind = intent_kind_of(&v);

    with_db_and_vec_write(|conn, vec| match db::ingest(conn, vec, &paths().bundle_dir, &bundle, &title, &text, &ext, &intent_kind, &tags, &links) {
        Ok((_id, slug, n)) => {
            let msg = format!("[OK] Ingestado '{filename}' en bundle '{bundle}' — {n} fragmento(s), id={slug}.");
            plugin_log(&msg);
            msg
        }
        Err(e) => {
            plugin_log(&format!("ERROR ingest_file '{filename}' bundle '{bundle}': {e}"));
            format!("[x] {e}")
        }
    })
    .unwrap_or_else(|e| e)
}

fn cmd_search(args: &str) -> String {
    let v = parse_args(args);
    let Some(query) = field_or_plain(&v, &["query", "consulta"]) else {
        return json!({ "ok": false, "error": "Escribí algo para buscar." }).to_string();
    };
    let query = query.to_string();
    let bundle = bundle_filter_of(&v);
    with_db_and_vec_read(|conn, vec| {
        let results = search::hybrid_search(conn, vec, &query, bundle.as_deref(), 6);
        // Se manda aparte del ranking en sí — la UI la puede mostrar como
        // "detectamos que buscás X" sin que cambie el cálculo del score.
        let detected_intent = search::classify_query_intent(&query);
        if results.is_empty() {
            return json!({ "ok": true, "results": [], "detected_intent": detected_intent })
                .to_string();
        }
        let results_json: Vec<Value> = results
            .iter()
            .map(|r| {
                let snippet: String = r
                    .text
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .chars()
                    .take(300)
                    .collect();
                json!({
                    "id": r.id,
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "slug": r.slug,
                    "bundle": r.bundle,
                    "okf_path": r.okf_path,
                    "intent_kind": r.intent_kind,
                    "score": r.score,
                    "snippet": snippet,
                    "text": r.text,
                })
            })
            .collect();
        json!({ "ok": true, "results": results_json, "detected_intent": detected_intent })
            .to_string()
    })
    .unwrap_or_else(|e| json!({ "ok": false, "error": e }).to_string())
}

fn cmd_list(args: &str) -> String {
    let v = parse_args(args);
    let bundle = bundle_filter_of(&v);
    // Tope duro: aunque alguien pida un límite más grande, nunca se
    // devuelven más de 2000 documentos de una — con millones de
    // documentos, un límite "sugerido" no alcanza si algo del lado de
    // la UI se olvida de mandarlo.
    let limit = v
        .get("limit")
        .and_then(|x| x.as_i64())
        .unwrap_or(200)
        .clamp(1, 2000);
    let offset = v.get("offset").and_then(|x| x.as_i64()).unwrap_or(0).max(0);
    with_db(
        |conn| match db::list_documents(conn, bundle.as_deref(), limit, offset) {
            Ok(docs) => serde_json::to_string(&docs).unwrap_or_else(|_| "[]".into()),
            Err(e) => format!("[x] {e}"),
        },
    )
    .unwrap_or_else(|e| e)
}

/// Conteo rápido (COUNT(*), no trae los documentos) — para el header de
/// la UI, que antes usaba rag_list y con millones de documentos eso
/// significaba serializar y mandar por IPC un JSON de cientos de MB
/// solo para contar cuántos había.
fn cmd_stats() -> String {
    with_db(|conn| {
        let documents: i64 = conn
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
            .unwrap_or(0);
        let chunks: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap_or(0);
        json!({ "documents": documents, "chunks": chunks }).to_string()
    })
    .unwrap_or_else(|e| e)
}

fn cmd_delete(args: &str) -> String {
    let v = parse_args(args);
    let Some(slug) = field_or_plain(&v, &["slug"]) else {
        return "[x] Falta el slug a borrar.".into();
    };
    let slug = slug.to_string();
    let bundle = match bundle_of(&v) {
        Ok(bundle) => bundle,
        Err(e) => return e,
    };
    with_db_and_vec_write(
        |conn, vec| match db::delete_document(conn, vec, &slug, &bundle) {
            Ok(Some(_removed_ids)) => {
                let msg = format!("[OK] Eliminado '{slug}' del bundle '{bundle}'.");
                plugin_log(&msg);
                msg
            }
            Ok(None) => format!("[x] No se encontró '{slug}' en el bundle '{bundle}'."),
            Err(e) => {
                plugin_log(&format!("ERROR delete '{slug}' bundle '{bundle}': {e}"));
                format!("[x] {e}")
            }
        },
    )
    .unwrap_or_else(|e| e)
}

/// Devuelve el contenido editable de un documento (título, texto sin el
/// heading, tags, links, source_kind) leyendo su .md de disco.
fn cmd_get_document(args: &str) -> String {
    let v = parse_args(args);
    let Some(slug) = field_or_plain(&v, &["slug"]) else {
        return json!({ "ok": false, "error": "Falta el slug." }).to_string();
    };
    let slug = slug.to_string();
    let bundle = match bundle_of(&v) {
        Ok(bundle) => bundle,
        Err(e) => return e,
    };
    with_db(|conn| match db::get_document(conn, &slug, &bundle) {
        Ok(Some(doc)) => json!({ "ok": true, "doc": doc }).to_string(),
        Ok(None) => json!({ "ok": false, "error": format!("No se encontró '{slug}' en el bundle '{bundle}'.") }).to_string(),
        Err(e) => json!({ "ok": false, "error": e }).to_string(),
    })
    .unwrap_or_else(|e| json!({ "ok": false, "error": e }).to_string())
}

/// Actualiza un documento existente in-place (reescribe el .md, re-chunkea
/// y re-embeddea SOLO ese documento) — no dispara ningún reindex/rebuild
/// del resto del índice.
fn cmd_update_document(args: &str) -> String {
    let v = parse_args(args);
    let Some(slug) = v
        .get("slug")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
    else {
        return "[x] Falta el slug a actualizar.".into();
    };
    let slug = slug.to_string();
    let bundle = match bundle_of(&v) {
        Ok(bundle) => bundle,
        Err(e) => return e,
    };
    let Some(text) = v
        .get("text")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
    else {
        return "[x] No hay texto para guardar.".into();
    };
    let title = v
        .get("title")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&slug)
        .to_string();
    let source_kind = v
        .get("source_kind")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("text")
        .to_string();
    let intent_kind = intent_kind_of(&v);
    let tags = str_list(&v, "tags");
    let links = str_list(&v, "links");

    with_db_and_vec_write(|conn, vec| {
        match db::update_document(
            conn,
            vec,
            &paths().bundle_dir,
            &bundle,
            &slug,
            &title,
            &text,
            &source_kind,
            &intent_kind,
            &tags,
            &links,
        ) {
            Ok(n) => {
                let msg =
                    format!("[OK] Actualizado '{slug}' en bundle '{bundle}' — {n} fragmento(s).");
                plugin_log(&msg);
                msg
            }
            Err(e) => {
                plugin_log(&format!(
                    "ERROR update_document '{slug}' bundle '{bundle}': {e}"
                ));
                format!("[x] {e}")
            }
        }
    })
    .unwrap_or_else(|e| e)
}

/// Margen conservador para el system prompt de Agentic (instrucciones,
/// herramientas y formato ChatML). El RAG no debe ocupar el contexto que
/// necesita el modelo para entender sus reglas ni para responder.
const SYSTEM_PROMPT_RESERVE_TOKENS: usize = 3_072;
const RAG_CONTEXT_SHARE_PERCENT: usize = 45;
const MIN_RAG_CONTEXT_CHARS: usize = 1_200;
const MAX_RAG_CONTEXT_CHARS: usize = 9_000;
const MAX_RETRIEVED_CHUNKS: usize = 6;

#[derive(Clone, Copy)]
struct RagContextBudget {
    n_ctx_tokens: usize,
    response_tokens: usize,
    chars: usize,
}

fn next_power_of_two(value: usize) -> usize {
    if value <= 1 {
        return 1;
    }
    value.checked_next_power_of_two().unwrap_or(usize::MAX)
}

/// Lee un entero simple de la configuración TOML sin duplicar el parser de
/// Agentic como dependencia del plugin RAG. Solo necesitamos dos claves
/// numéricas estables: `n_ctx` y `max_tokens`.
fn toml_usize(raw: &str, key: &str) -> Option<usize> {
    raw.lines().find_map(|line| {
        let line = line.split('#').next()?.trim();
        let (name, value) = line.split_once('=')?;
        (name.trim() == key).then(|| value.trim().trim_matches('"').parse().ok())?
    })
}

fn agentic_config_path() -> PathBuf {
    if let Ok(path) = std::env::var("AME_AGENTIC_CONFIG") {
        return PathBuf::from(path);
    }

    // Knowledge = <raíz de AME>/Brain/Memory/Knowledge; por eso subir dos
    // niveles nos deja en Brain y permite encontrar Config sin hardcodear D:.
    paths()
        .knowledge_dir
        .parent()
        .and_then(|memory| memory.parent())
        .map(|brain| brain.join("Config").join("llm_inference_hyperparameters.toml"))
        .unwrap_or_else(|| PathBuf::from("Brain/Config/llm_inference_hyperparameters.toml"))
}

/// Replica la parte relevante del cálculo de Agentic para decidir cuánto
/// contexto RAG cabe. Se calcula en cada consulta: si el usuario cambia
/// `n_ctx` o `max_tokens` en Agentic, el RAG se adapta sin reiniciar ni
/// reindexar los documentos.
fn rag_context_budget_from_values(
    configured_ctx: usize,
    configured_max_tokens: usize,
    query_chars: usize,
) -> RagContextBudget {
    // Misma reserva mínima que Agentic: el contexto efectivo puede crecer
    // al siguiente múltiplo de dos si `n_ctx` no alcanza para su prompt.
    let minimum_ctx = SYSTEM_PROMPT_RESERVE_TOKENS + 512 * 4 + 400;
    let n_ctx_tokens = if configured_ctx >= minimum_ctx {
        next_power_of_two(configured_ctx)
    } else {
        next_power_of_two(minimum_ctx)
    };
    let response_tokens = if configured_max_tokens > 0 {
        configured_max_tokens.min(n_ctx_tokens / 2)
    } else {
        (n_ctx_tokens / 4).clamp(256, 2_048)
    };

    // El historial puede crecer durante una conversación. Reservamos 512
    // tokens extra y solo usamos 45% de lo que queda para RAG, en vez de
    // llenar toda la ventana con texto recuperado.
    let query_tokens = query_chars.saturating_add(2) / 3;
    let free_tokens = n_ctx_tokens
        .saturating_sub(response_tokens)
        .saturating_sub(SYSTEM_PROMPT_RESERVE_TOKENS)
        .saturating_sub(query_tokens)
        .saturating_sub(512);
    let rag_tokens = free_tokens.saturating_mul(RAG_CONTEXT_SHARE_PERCENT) / 100;
    let chars = rag_tokens
        .saturating_mul(3)
        .clamp(MIN_RAG_CONTEXT_CHARS, MAX_RAG_CONTEXT_CHARS);

    RagContextBudget {
        n_ctx_tokens,
        response_tokens,
        chars,
    }
}

fn rag_context_budget(query: &str) -> RagContextBudget {
    let raw = std::fs::read_to_string(agentic_config_path()).unwrap_or_default();
    let configured_ctx = toml_usize(&raw, "n_ctx").unwrap_or(4_096);
    let configured_max_tokens = toml_usize(&raw, "max_tokens").unwrap_or(0);
    rag_context_budget_from_values(configured_ctx, configured_max_tokens, query.chars().count())
}

fn cmd_on_before_request(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }
    let t0 = Instant::now();
    let mut top_score: Option<f64> = None;
    let mut discarded_low_confidence = false;
    let mut injected_chunks = 0usize;
    let budget = rag_context_budget(text);
    let result = with_db_and_vec_read(|conn, vec| {
        // Recuperamos fragmentos, no documentos completos. `chunks` ya se
        // genera al ingerir (máx. ~900 caracteres), así que una pregunta
        // sobre un canto de La Ilíada recibe pasajes relevantes y nunca el
        // PDF entero en un único bloque.
        let results = search::hybrid_search(conn, vec, text, None, MAX_RETRIEVED_CHUNKS);
        if results.is_empty() {
            return text.to_string();
        }

        top_score = results.first().map(|r| r.score);
        // Vara extra, más alta que la de `hybrid_search` en sí, antes de
        // inyectar en el prompt — ver comment de `INJECT_MIN_SCORE`.
        // Esto es lo que evita que un match débil (que sí sirve para
        // aparecer en una lista de búsqueda) se cuele como "contexto
        // recuperado" y el modelo termine respondiendo con algo que no
        // tiene nada que ver con la pregunta.
        if top_score.unwrap_or(0.0) < search::INJECT_MIN_SCORE {
            discarded_low_confidence = true;
            return text.to_string();
        }

        let mut lines = vec![
            text.to_string(),
            String::new(),
            "[Contexto recuperado — base de conocimiento local (RAG)]\nMaterial de referencia para responder la pregunta anterior. Úsalo para redactar una respuesta propia y directa; no lo copies ni lo reproduzcas salvo que el usuario pida el texto original. Si no contiene la respuesta, responde con tu conocimiento general e indica que el contexto no la incluye.".to_string(),
        ];

        let mut used_chars = 0usize;
        for r in &results {
            if injected_chunks >= MAX_RETRIEVED_CHUNKS {
                break;
            }
            let separator_chars = if injected_chunks == 0 { 0 } else { 2 };
            let label = format!("({}) {} [{}]", injected_chunks + 1, r.title, r.bundle);
            let framing_chars = label.chars().count() + 2; // ":\n"
            let remaining = budget
                .chars
                .saturating_sub(used_chars.saturating_add(separator_chars).saturating_add(framing_chars));
            if remaining < 160 {
                break;
            }

            // La condensación es una segunda barrera: también protege
            // índices creados por versiones antiguas con un chunk anómalo.
            let content = qa::condense(text, &r.text, remaining);
            if content.trim().is_empty() {
                continue;
            }
            let part = format!("{label}:\n{content}");
            used_chars = used_chars
                .saturating_add(separator_chars)
                .saturating_add(part.chars().count());
            lines.push(part);
            injected_chunks += 1;
        }
        if injected_chunks == 0 {
            text.to_string()
        } else {
            lines.join("\n\n")
        }
    })
    .unwrap_or_else(|_| text.to_string());
    let elapsed_ms = t0.elapsed().as_millis();
    let score_note = match (top_score, discarded_low_confidence) {
        (Some(s), true) => format!(" score={s:.3} (descartado, < INJECT_MIN_SCORE)"),
        (Some(s), false) => format!(" score={s:.3}"),
        (None, _) => String::new(),
    };
    plugin_log(&format!(
        "on_before_request: {elapsed_ms}ms{score_note} | chunks={injected_chunks} budget={} chars n_ctx={} respuesta={} | para \"{}\"",
        budget.chars,
        budget.n_ctx_tokens,
        budget.response_tokens,
        text.chars().take(60).collect::<String>(),
    ));
    result
}

fn cmd_okf_graph(args: &str) -> String {
    let v = parse_args(args);
    let bundle = bundle_filter_of(&v);
    // `sim`: calcular aristas por similitud (la UI lo pide solo si el
    // checkbox "Similitud" está marcado). `max_nodes`: cuántos nodos
    // (los más conectados) se envían al navegador.
    let opts = graph::GraphOpts {
        sim: v.get("sim").and_then(|x| x.as_bool()).unwrap_or(false),
        max_nodes: v
            .get("max_nodes")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(graph::DEFAULT_MAX_NODES),
    };
    with_db(
        |conn| match graph::build_okf_graph(conn, bundle.as_deref(), opts) {
            Ok(g) => serde_json::to_string(&g).unwrap_or_else(|_| "{}".into()),
            Err(e) => format!("[x] {e}"),
        },
    )
    .unwrap_or_else(|e| e)
}

fn cmd_similarity_map(args: &str) -> String {
    let v = parse_args(args);
    let query = field_or_plain(&v, &["query", "consulta"])
        .unwrap_or("")
        .to_string();
    let bundle = bundle_filter_of(&v);
    with_db_and_vec_read(|conn, vec| {
        match graph::build_similarity_map(conn, vec, bundle.as_deref(), &query) {
            Ok(m) => serde_json::to_string(&m).unwrap_or_else(|_| "{}".into()),
            Err(e) => format!("[x] {e}"),
        }
    })
    .unwrap_or_else(|e| e)
}

fn cmd_reindex(args: &str) -> String {
    let v = parse_args(args);
    let Some(bundle) = field_or_plain(&v, &["bundle"]) else {
        return "[x] Indicá qué bundle reindexar (ej: rag_reindex wikipedia).".into();
    };
    let bundle = bundle.to_string();
    if !frontmatter::valid_bundle_name(&bundle) {
        return "[x] Nombre de bundle inválido — solo letras, números, '-' y '_' (sin espacios ni símbolos).".into();
    }
    if let Err(e) = try_start_job(&format!("reindex de '{bundle}'")) {
        return e;
    }
    plugin_log(&format!("reindex de '{bundle}' iniciado en background"));
    std::thread::spawn(move || {
        let outcome = with_db_and_vec_write(|conn, vec| {
            reindex::sync_from_bundle_parallel(
                conn,
                vec,
                &paths().bundle_dir,
                &bundle,
                &JOB_PROGRESS,
                &JOB_TOTAL,
            )
        });
        let msg = match outcome {
            Ok(Ok(stats)) => {
                let m = format!(
                    "[OK] Reindex de '{bundle}': {} escaneados, {} nuevos, {} actualizados, {} sin cambios, {} eliminados, {} errores.",
                    stats.scanned, stats.added, stats.updated, stats.unchanged, stats.removed, stats.errors
                );
                plugin_log(&m);
                m
            }
            Ok(Err(e)) => {
                plugin_log(&format!("ERROR reindex '{bundle}': {e}"));
                format!("[x] {e}")
            }
            Err(e) => {
                plugin_log(&format!("ERROR reindex '{bundle}' (host no listo): {e}"));
                format!("[x] {e}")
            }
        };
        finish_job(msg);
    });
    "[OK] Reindex iniciado en background — la app no se va a congelar. Consultá el progreso con rag_job_status.".to_string()
}

fn cmd_rebuild_index() -> String {
    if let Err(e) = try_start_job("reconstrucción completa del índice") {
        return e;
    }
    plugin_log("rebuild_index iniciado en background");
    std::thread::spawn(|| {
        let outcome = with_db_and_vec_write(|conn, vec| {
            reindex::rebuild_index_parallel(
                conn,
                vec,
                &paths().bundle_dir,
                &JOB_PROGRESS,
                &JOB_TOTAL,
            )
        });
        let msg = match outcome {
            Ok(Ok(stats)) => {
                let m = format!(
                    "[OK] Índice reconstruido desde los .md: {} bundles, {} documentos, {} fragmentos, {} errores.",
                    stats.bundles, stats.documents, stats.chunks, stats.errors
                );
                plugin_log(&m);
                m
            }
            Ok(Err(e)) => {
                plugin_log(&format!("ERROR rebuild_index: {e}"));
                format!("[x] {e}")
            }
            Err(e) => {
                plugin_log(&format!("ERROR rebuild_index (host no listo): {e}"));
                format!("[x] {e}")
            }
        };
        finish_job(msg);
    });
    "[OK] Reconstrucción iniciada en background — la app no se va a congelar. Consultá el progreso con rag_job_status.".to_string()
}

fn cmd_list_bundles() -> String {
    with_db(|conn| {
        serde_json::to_string(&db::list_bundles(conn, &paths().bundle_dir))
            .unwrap_or_else(|_| "[]".into())
    })
    .unwrap_or_else(|e| e)
}

fn cmd_create_bundle(args: &str) -> String {
    let v = parse_args(args);
    let Some(name) = field_or_plain(&v, &["name", "bundle"]) else {
        return "[x] Falta el nombre del bundle.".into();
    };
    let name = name.to_string();
    if !frontmatter::valid_bundle_name(&name) {
        return "[x] Nombre inválido — solo letras, números, '-' y '_' (sin espacios ni símbolos)."
            .into();
    }
    if let Err(e) = ensure_ready() {
        return e;
    }
    match db::create_bundle(&paths().bundle_dir, &name) {
        Ok(()) => {
            let msg = format!("[OK] Bundle '{name}' creado.");
            plugin_log(&msg);
            msg
        }
        Err(e) => {
            plugin_log(&format!("ERROR create_bundle '{name}': {e}"));
            format!("[x] {e}")
        }
    }
}

fn cmd_rename_bundle(args: &str) -> String {
    let v = parse_args(args);
    let Some(old) = v
        .get("old")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
    else {
        return "[x] Falta el bundle actual ('old').".into();
    };
    let Some(new) = v
        .get("new")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
    else {
        return "[x] Falta el nuevo nombre ('new').".into();
    };
    if !frontmatter::valid_bundle_name(old) || !frontmatter::valid_bundle_name(new) {
        return "[x] Nombre inválido — solo letras, números, '-' y '_' (sin espacios ni símbolos)."
            .into();
    }
    if old == "default" {
        return "[x] No se puede renombrar el bundle 'default'.".into();
    }
    let (old, new) = (old.to_string(), new.to_string());
    with_db(
        |conn| match db::rename_bundle(conn, &paths().bundle_dir, &old, &new) {
            Ok(n) => {
                let msg = format!(
                    "[OK] Bundle '{old}' renombrado a '{new}' ({n} documento(s) actualizados)."
                );
                plugin_log(&msg);
                msg
            }
            Err(e) => {
                plugin_log(&format!("ERROR rename_bundle '{old}' -> '{new}': {e}"));
                format!("[x] {e}")
            }
        },
    )
    .unwrap_or_else(|e| e)
}

fn cmd_delete_bundle(args: &str) -> String {
    let v = parse_args(args);
    let Some(name) = field_or_plain(&v, &["name", "bundle"]) else {
        return "[x] Falta el nombre del bundle a borrar.".into();
    };
    let name = name.to_string();
    if !frontmatter::valid_bundle_name(&name) {
        return "[x] Nombre de bundle inválido — solo letras, números, '-' y '_' (sin espacios ni símbolos).".into();
    }
    if name == "default" {
        return "[x] No se puede borrar el bundle 'default'.".into();
    }
    with_db_and_vec_write(|conn, vec| match db::delete_bundle(conn, vec, &paths().bundle_dir, &name) {
        Ok(n) => {
            let msg = format!("[OK] Bundle '{name}' eliminado — {n} documento(s) borrados junto con la carpeta.");
            plugin_log(&msg);
            msg
        }
        Err(e) => {
            plugin_log(&format!("ERROR delete_bundle '{name}': {e}"));
            format!("[x] {e}")
        }
    })
    .unwrap_or_else(|e| e)
}

fn dispatch(cmd: &str, args: &str) -> String {
    match cmd {
        "rag_ingest_text" => cmd_ingest_text(args),
        "rag_ingest_file" => cmd_ingest_file(args),
        "rag_search" => cmd_search(args),
        "rag_list" => cmd_list(args),
        "rag_stats" => cmd_stats(),
        "rag_delete" => cmd_delete(args),
        "rag_get_document" => cmd_get_document(args),
        "rag_update_document" => cmd_update_document(args),
        "rag_okf_graph" => cmd_okf_graph(args),
        "rag_similarity_map" => cmd_similarity_map(args),
        "rag_reindex" => cmd_reindex(args),
        "rag_rebuild_index" => cmd_rebuild_index(),
        "rag_list_bundles" => cmd_list_bundles(),
        "rag_create_bundle" => cmd_create_bundle(args),
        "rag_rename_bundle" => cmd_rename_bundle(args),
        "rag_delete_bundle" => cmd_delete_bundle(args),
        "rag_job_status" => cmd_job_status(),
        "rag_db_stats" => cmd_db_stats(),
        "rag_wal_checkpoint" => cmd_wal_checkpoint(),
        "rag_vacuum" => cmd_vacuum(),
        "on_before_request" => cmd_on_before_request(args),
        _ => format!("[x] Comando desconocido: {cmd}"),
    }
}

fn command_defs() -> Value {
    let arg =
        |name: &str, required: bool| json!({"name": name, "type": "string", "required": required});
    json!([
        {"name": "rag_ingest_text", "alias": [], "category": "RAG",
         "desc": "Ingesta texto pegado: JSON {title,text,tags,links,bundle,intent_kind}. intent_kind opcional: 'info'/'herramienta'/'plan'.", "args": [arg("json", true)]},
        {"name": "rag_ingest_file", "alias": [], "category": "RAG",
         "desc": "Ingesta un archivo (.txt/.md/.pdf — PDF escaneado sin capa de texto no funciona): JSON {filename,content_b64,tags,links,bundle,intent_kind}.",
         "args": [arg("json", true)]},
        {"name": "rag_search", "alias": ["buscar_rag"], "category": "RAG",
         "desc": "rag_search <consulta> o JSON {query,bundle}. La respuesta incluye detected_intent ('info'/'herramienta'/'plan'/null).", "args": [arg("consulta", true)]},
        {"name": "rag_list", "alias": [], "category": "RAG",
         "desc": "Lista documentos, paginado. Args JSON: bundle (opcional), limit (default 200, tope 2000), offset (default 0).", "args": []},
        {"name": "rag_stats", "alias": [], "category": "RAG",
         "desc": "Conteo rápido de documentos/fragmentos (JSON {documents, chunks}), sin traer los documentos.", "args": []},
        {"name": "rag_delete", "alias": [], "category": "RAG",
         "desc": "rag_delete <slug> o JSON {slug,bundle}.", "args": [arg("slug", true)]},
        {"name": "rag_get_document", "alias": [], "category": "RAG",
         "desc": "rag_get_document JSON {slug,bundle} — devuelve el contenido editable de un documento (title,text,tags,links,source_kind).", "args": [arg("json", true)]},
        {"name": "rag_update_document", "alias": [], "category": "RAG",
         "desc": "Actualiza un documento existente sin reindexar el resto: JSON {slug,bundle,title,text,tags,links,source_kind,intent_kind}.", "args": [arg("json", true)]},
        {"name": "rag_okf_graph", "alias": [], "category": "RAG",
         "desc": "Grafo de relaciones (links + tags + similitud). Args: bundle opcional.", "args": []},
        {"name": "rag_similarity_map", "alias": [], "category": "RAG",
         "desc": "Mapa 2D (PCA) de fragmentos. Args: JSON {query,bundle}, ambos opcionales.", "args": []},
        {"name": "rag_reindex", "alias": [], "category": "RAG",
         "desc": "rag_reindex <bundle> — sincroniza SQLite con los .md de ese bundle por hash.", "args": [arg("bundle", true)]},
        {"name": "rag_rebuild_index", "alias": [], "category": "RAG",
         "desc": "Reconstruye TODO el índice SQLite desde los .md de todos los bundles (destructivo sobre el .sqlite3, no sobre los .md).", "args": []},
        {"name": "rag_list_bundles", "alias": [], "category": "RAG",
         "desc": "Lista los bundles existentes (carpetas en disco + valores en SQLite).", "args": []},
        {"name": "rag_create_bundle", "alias": [], "category": "RAG",
         "desc": "rag_create_bundle <nombre> — crea una carpeta de bundle nueva y vacía.", "args": [arg("nombre", true)]},
        {"name": "rag_rename_bundle", "alias": [], "category": "RAG",
         "desc": "Renombra un bundle: JSON {old,new}. Mueve la carpeta y actualiza SQLite.", "args": [arg("json", true)]},
        {"name": "rag_delete_bundle", "alias": [], "category": "RAG",
         "desc": "rag_delete_bundle <nombre> — borra TODOS los documentos de ese bundle y su carpeta en disco.", "args": [arg("nombre", true)]},
        {"name": "rag_job_status", "alias": [], "category": "RAG",
         "desc": "Progreso del reindex/rebuild/vacuum en curso (JSON {running, documents_processed, last_result}).", "args": []},
        {"name": "rag_db_stats", "alias": [], "category": "RAG",
         "desc": "Tamaño real del .sqlite3 vs. espacio reciclable (freelist) — para decidir si conviene VACUUM.", "args": []},
        {"name": "rag_wal_checkpoint", "alias": [], "category": "RAG",
         "desc": "Vuelca el .sqlite3-wal al archivo principal y lo achica — rápido, no reclama espacio del archivo principal.", "args": []},
        {"name": "rag_vacuum", "alias": [], "category": "RAG",
         "desc": "Reconstruye el .sqlite3 entero sin espacio libre sobrante — lento en archivos grandes, corre en background.", "args": []},
        {"name": "on_before_request", "alias": [], "category": "hook",
         "desc": "(hook interno — inyecta contexto recuperado antes de cada mensaje)", "args": []},
    ])
}

// ═══════════════════════════════════════════════════════════════════════
// UI embebida — en release queda DENTRO del .dll (un solo archivo para
// distribuir/actualizar en vez de .dll + ui.html sueltos); en debug se
// sigue leyendo de disco en cada llamada para poder iterar el HTML del
// chat sin recompilar. Ver ame_plugin_ui() más abajo.
//
// IMPORTANTE:
// La carpeta `ui/` debe estar en la raíz del crate, al mismo nivel que
// Cargo.toml:
//
//   tu_plugin/
//   ├─ Cargo.toml
//   ├─ ui/
//   │  └─ ui.html
//   └─ src/
//      └─ lib.rs
// ═══════════════════════════════════════════════════════════════════════

#[cfg(not(debug_assertions))]
#[derive(rust_embed::RustEmbed)]
#[folder = "ui/"]
struct UiAssets;

fn ui_html() -> Option<String> {
    #[cfg(debug_assertions)]
    {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("ui")
            .join("ui.html");

        match std::fs::read_to_string(&path) {
            Ok(html) => Some(html),
            Err(e) => {
                plugin_log(&format!(
                    "ame_plugin_ui: no pude leer ui.html desde disco ({}): {e}",
                    path.display()
                ));
                None
            }
        }
    }

    #[cfg(not(debug_assertions))]
    {
        UiAssets::get("ui.html")
            .map(|file| String::from_utf8_lossy(file.data.as_ref()).into_owned())
    }
}

// ─── ABI de C — ver cmd_registry.rs del host ─────────────────────────────
fn to_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => CString::new("[x] respuesta con NUL embebido")
            .unwrap()
            .into_raw(),
    }
}

#[no_mangle]
pub extern "C" fn ame_plugin_info() -> *mut c_char {
    to_c_string(command_defs().to_string())
}

#[no_mangle]
pub extern "C" fn ame_plugin_ui() -> *mut c_char {
    plugin_log("ame_plugin_ui() llamado");

    match ui_html() {
        Some(html) => {
            plugin_log(&format!("  -> ui.html servido ({} bytes)", html.len()));
            to_c_string(html)
        }
        None => {
            plugin_log("ame_plugin_ui: ui.html no encontrado");
            to_c_string(String::new())
        }
    }
}

/// # Safety
/// `name`/`args` deben ser punteros C válidos terminados en NUL.
#[no_mangle]
pub unsafe extern "C" fn ame_plugin_execute(
    name: *const c_char,
    args: *const c_char,
) -> *mut c_char {
    if name.is_null() || args.is_null() {
        return to_c_string("[x] argumentos nulos".into());
    }
    let name = CStr::from_ptr(name).to_string_lossy().into_owned();
    let args = CStr::from_ptr(args).to_string_lossy().into_owned();
    let cmd_for_log = name.clone();
    let result = std::panic::catch_unwind(move || dispatch(&name, &args)).unwrap_or_else(|_| {
        plugin_log(&format!("PÁNICO ejecutando \"{cmd_for_log}\""));
        format!("[x] el plugin RAG paniqueó procesando el comando \"{cmd_for_log}\"")
    });
    to_c_string(result)
}

/// # Safety
/// `ptr` debe venir de `ame_plugin_info`/`_execute` de ESTE plugin.
#[no_mangle]
pub unsafe extern "C" fn ame_plugin_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

#[no_mangle]
pub extern "C" fn ame_plugin_name() -> *mut c_char {
    to_c_string("RAG".into())
}

#[cfg(test)]
mod tests {
    use super::rag_context_budget_from_values;

    #[test]
    fn rag_budget_grows_with_context_window_but_stays_bounded() {
        let small = rag_context_budget_from_values(4_096, 0, 60);
        let large = rag_context_budget_from_values(16_384, 0, 60);

        // Agentic eleva 4096 a 8192 para que su prompt base quepa.
        assert_eq!(small.n_ctx_tokens, 8_192);
        assert_eq!(small.response_tokens, 2_048);
        assert!(small.chars >= 1_200);
        assert!(large.chars > small.chars);
        assert!(large.chars <= 9_000);
    }
}