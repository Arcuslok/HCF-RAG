//! db.rs — esquema SQLite (texto + metadata) e ingesta OKF con bundles.
//!
//! Los vectores YA NO viven acá — ver vec_index.rs. Este archivo solo
//! guarda `documents`/`chunks` (texto) + FTS5 (léxico). La key de cada
//! chunk en el índice de vectores es directamente `chunks.id`: se llama
//! a `vec_index.add()` justo después del INSERT, dentro de la misma
//! función, para que los dos lados queden sincronizados por
//! construcción — nunca hay un id de más de un lado sin el otro.

use rusqlite::{params, Connection};
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::embed::{chunk_text, embed};
use crate::frontmatter;
use crate::vec_index::VecIndex;

pub fn open(db_path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch(
        "
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS documents (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            slug         TEXT NOT NULL,
            bundle       TEXT NOT NULL DEFAULT 'default',
            title        TEXT NOT NULL,
            source_kind  TEXT NOT NULL,
            intent_kind  TEXT NOT NULL DEFAULT '',
            okf_path     TEXT NOT NULL,
            tags         TEXT NOT NULL DEFAULT '[]',
            links        TEXT NOT NULL DEFAULT '[]',
            content_hash TEXT NOT NULL DEFAULT '',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            UNIQUE(bundle, slug)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index  INTEGER NOT NULL,
            text         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_documents_bundle ON documents(bundle);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, content='chunks', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        END;
        ",
    )?;
    migrate_intent_kind(&conn)?;
    Ok(conn)
}

/// Agrega la columna `intent_kind` a bases existentes creadas antes de
/// este campo — `CREATE TABLE IF NOT EXISTS` de arriba no toca una
/// tabla que ya existe, así que una base vieja nunca vería la columna
/// nueva sin este paso. Chequea con `pragma_table_info` en vez de
/// intentar el ALTER a ciegas (que fallaría con "duplicate column" en
/// una base ya migrada).
fn migrate_intent_kind(conn: &Connection) -> rusqlite::Result<()> {
    let has_col: bool = {
        let mut stmt =
            conn.prepare("SELECT 1 FROM pragma_table_info('documents') WHERE name='intent_kind'")?;
        stmt.exists([])?
    };
    if !has_col {
        conn.execute_batch(
            "ALTER TABLE documents ADD COLUMN intent_kind TEXT NOT NULL DEFAULT '';",
        )?;
    }
    Ok(())
}

fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m2 = if mp < 10 { mp + 3 } else { mp - 9 };
    let y2 = if m2 <= 2 { y + 1 } else { y };
    format!("{y2:04}-{m2:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn unique_slug(conn: &Connection, base: &str, bundle: &str) -> rusqlite::Result<String> {
    let mut slug = base.to_string();
    let mut i = 2;
    loop {
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM documents WHERE slug=?1 AND bundle=?2",
                params![slug, bundle],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !exists {
            return Ok(slug);
        }
        slug = format!("{base}-{i}");
        i += 1;
    }
}

fn insert_chunks(
    conn: &Connection,
    vec_index: &mut VecIndex,
    doc_id: i64,
    text: &str,
    bundle: &str,
) -> Result<usize, String> {
    let chunks = chunk_text(text);
    let mut ids: Vec<i64> = Vec::with_capacity(chunks.len());
    let mut flat: Vec<f32> = Vec::with_capacity(chunks.len() * crate::vec_index::DIMENSIONS);
    for (i, ch) in chunks.iter().enumerate() {
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text) VALUES (?1,?2,?3)",
            params![doc_id, i as i64, ch],
        )
        .map_err(|e| e.to_string())?;
        ids.push(conn.last_insert_rowid());
        flat.extend_from_slice(&embed(ch));
    }
    vec_index.add_batch(bundle, &ids, &flat)?;
    Ok(chunks.len())
}

/// Igual que `insert_chunks` pero con los embeddings YA calculados — la
/// contraparte "solo escritura" para el pipeline paralelo (ver
/// reindex.rs `prepare_document`): el cálculo pesado se hizo en un hilo
/// de trabajo aparte, este hilo (el único que toca las conexiones) solo
/// hace el INSERT + el add() al índice de vectores.
fn insert_chunks_precomputed(
    conn: &Connection,
    vec_index: &mut VecIndex,
    doc_id: i64,
    chunks: &[(String, Vec<f32>)],
    bundle: &str,
) -> Result<usize, String> {
    let mut ids: Vec<i64> = Vec::with_capacity(chunks.len());
    let mut flat: Vec<f32> = Vec::with_capacity(chunks.len() * crate::vec_index::DIMENSIONS);
    for (i, (text, emb)) in chunks.iter().enumerate() {
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text) VALUES (?1,?2,?3)",
            params![doc_id, i as i64, text],
        )
        .map_err(|e| e.to_string())?;
        ids.push(conn.last_insert_rowid());
        flat.extend_from_slice(emb);
    }
    vec_index.add_batch(bundle, &ids, &flat)?;
    Ok(chunks.len())
}

/// Ingesta "en caliente" (texto pegado por el usuario o un archivo
/// subido) — escribe el .md, inserta en `documents`+`chunks`+vec_index.
pub fn ingest(
    conn: &Connection,
    vec_index: &mut VecIndex,
    bundle_dir: &Path,
    bundle: &str,
    title: &str,
    text: &str,
    source_kind: &str,
    intent_kind: &str,
    tags: &[String],
    links: &[String],
) -> Result<(i64, String, usize), String> {
    let slug =
        unique_slug(conn, &frontmatter::slugify(title), bundle).map_err(|e| e.to_string())?;
    let now = now_iso();
    let (okf_path, content_hash) = frontmatter::write_okf(
        bundle_dir,
        bundle,
        &slug,
        title,
        text,
        source_kind,
        intent_kind,
        tags,
        links,
        &now,
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "INSERT INTO documents (slug, bundle, title, source_kind, intent_kind, okf_path, tags, links, content_hash, created_at, updated_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?10)",
        params![
            slug, bundle, title, source_kind, intent_kind, okf_path.to_string_lossy(),
            serde_json::to_string(tags).unwrap(), serde_json::to_string(links).unwrap(), content_hash, now
        ],
    )
    .map_err(|e| e.to_string())?;
    let doc_id = conn.last_insert_rowid();
    let n = insert_chunks(conn, vec_index, doc_id, text, bundle)?;
    Ok((doc_id, slug, n))
}

/// Resultado de preparar un documento fuera del hilo escritor: parseo +
/// chunking + embeddings ya resueltos (floats crudos — ya no hace falta
/// serializarlos a bytes, no van a un BLOB de SQLite), listo para
/// insertar con una sola pasada.
pub struct PreparedInsert {
    pub slug: String,
    pub bundle: String,
    pub title: String,
    pub source_kind: String,
    pub intent_kind: String,
    pub tags: Vec<String>,
    pub links: Vec<String>,
    pub content_hash: String,
    pub okf_path: PathBuf,
    pub chunks: Vec<(String, Vec<f32>)>,
}

/// Contraparte de `ingest_from_okf` que recibe el trabajo pesado ya
/// hecho — reemplaza el documento anterior si existía (DELETE +
/// re-INSERT, podando también sus vectores del índice) y solo escribe.
pub fn insert_prepared(
    conn: &Connection,
    vec_index: &mut VecIndex,
    p: &PreparedInsert,
) -> Result<(i64, usize), String> {
    let old_doc_id: Option<i64> = conn
        .query_row(
            "SELECT id FROM documents WHERE slug=?1 AND bundle=?2",
            params![p.slug, p.bundle],
            |r| r.get(0),
        )
        .ok();
    if let Some(old_id) = old_doc_id {
        let mut stmt = conn
            .prepare("SELECT id FROM chunks WHERE doc_id=?1")
            .map_err(|e| e.to_string())?;
        let old_chunk_ids: Vec<i64> = stmt
            .query_map(params![old_id], |r| r.get(0))
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .collect();
        drop(stmt);
        for cid in old_chunk_ids {
            vec_index.remove(&p.bundle, cid);
        }
        conn.execute("DELETE FROM documents WHERE id=?1", params![old_id])
            .map_err(|e| e.to_string())?;
    }
    let now = now_iso();
    conn.execute(
        "INSERT INTO documents (slug, bundle, title, source_kind, intent_kind, okf_path, tags, links, content_hash, created_at, updated_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?10)",
        params![
            p.slug, p.bundle, p.title, p.source_kind, p.intent_kind, p.okf_path.to_string_lossy(),
            serde_json::to_string(&p.tags).unwrap(), serde_json::to_string(&p.links).unwrap(), p.content_hash, now
        ],
    )
    .map_err(|e| e.to_string())?;
    let doc_id = conn.last_insert_rowid();
    let n = insert_chunks_precomputed(conn, vec_index, doc_id, &p.chunks, &p.bundle)?;
    Ok((doc_id, n))
}

#[derive(Serialize)]
pub struct DocContent {
    pub slug: String,
    pub bundle: String,
    pub title: String,
    pub source_kind: String,
    pub intent_kind: String,
    pub tags: Vec<String>,
    pub links: Vec<String>,
    pub text: String,
}

/// Lee el .md de disco (fuente de verdad) y lo parsea para edición —
/// separa el heading `# título` del cuerpo real vía
/// `frontmatter::split_heading`, así el textarea de la UI muestra solo
/// el texto, no el markdown completo del archivo.
pub fn get_document(
    conn: &Connection,
    slug: &str,
    bundle: &str,
) -> Result<Option<DocContent>, String> {
    let okf_path: Option<String> = conn
        .query_row(
            "SELECT okf_path FROM documents WHERE slug=?1 AND bundle=?2",
            params![slug, bundle],
            |r| r.get(0),
        )
        .ok();
    let Some(okf_path) = okf_path else {
        return Ok(None);
    };
    let raw = std::fs::read_to_string(&okf_path).map_err(|e| e.to_string())?;
    let parsed = frontmatter::parse_okf(&raw);
    let title = if parsed.title.is_empty() {
        slug.to_string()
    } else {
        parsed.title
    };
    let text = frontmatter::split_heading(&parsed.body, &title);
    Ok(Some(DocContent {
        slug: slug.to_string(),
        bundle: bundle.to_string(),
        title,
        source_kind: parsed.source_kind,
        intent_kind: parsed.intent_kind,
        tags: parsed.tags,
        links: parsed.links,
        text,
    }))
}

/// Actualiza un documento existente: reescribe su .md (misma slug, mismo
/// path) y REEMPLAZA solo su fila + sus chunks + sus vectores — el resto
/// del índice (todos los demás documentos) queda intacto. A diferencia
/// de `rag_reindex`/`rag_rebuild_index`, esto no vuelve a leer ni
/// re-embeddear ningún otro archivo del bundle.
pub fn update_document(
    conn: &Connection,
    vec_index: &mut VecIndex,
    bundle_dir: &Path,
    bundle: &str,
    slug: &str,
    title: &str,
    text: &str,
    source_kind: &str,
    intent_kind: &str,
    tags: &[String],
    links: &[String],
) -> Result<usize, String> {
    let old_row: Option<(i64, String)> = conn
        .query_row(
            "SELECT id, created_at FROM documents WHERE slug=?1 AND bundle=?2",
            params![slug, bundle],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    let Some((old_id, created_at)) = old_row else {
        return Err(format!(
            "No se encontró '{slug}' en el bundle '{bundle}' para actualizar."
        ));
    };

    let now = now_iso();
    let (okf_path, content_hash) = frontmatter::write_okf(
        bundle_dir,
        bundle,
        slug,
        title,
        text,
        source_kind,
        intent_kind,
        tags,
        links,
        &now,
    )
    .map_err(|e| e.to_string())?;

    let mut stmt = conn
        .prepare("SELECT id FROM chunks WHERE doc_id=?1")
        .map_err(|e| e.to_string())?;
    let old_chunk_ids: Vec<i64> = stmt
        .query_map(params![old_id], |r| r.get(0))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();
    drop(stmt);
    for cid in old_chunk_ids {
        vec_index.remove(bundle, cid);
    }
    conn.execute("DELETE FROM documents WHERE id=?1", params![old_id])
        .map_err(|e| e.to_string())?;

    conn.execute(
        "INSERT INTO documents (slug, bundle, title, source_kind, intent_kind, okf_path, tags, links, content_hash, created_at, updated_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
        params![
            slug, bundle, title, source_kind, intent_kind, okf_path.to_string_lossy(),
            serde_json::to_string(tags).unwrap(), serde_json::to_string(links).unwrap(), content_hash, created_at, now
        ],
    )
    .map_err(|e| e.to_string())?;
    let doc_id = conn.last_insert_rowid();
    insert_chunks(conn, vec_index, doc_id, text, bundle)
}

#[derive(Serialize)]
pub struct DocInfo {
    pub slug: String,
    pub bundle: String,
    pub title: String,
    pub source_kind: String,
    pub intent_kind: String,
    pub tags: Vec<String>,
    pub links: Vec<String>,
    pub created_at: String,
    pub chunks: i64,
}

pub fn list_documents(
    conn: &Connection,
    bundle: Option<&str>,
    limit: i64,
    offset: i64,
) -> rusqlite::Result<Vec<DocInfo>> {
    // ORDER BY d.id DESC en vez de created_at: id ya es el orden de
    // inserción y es la primary key (rowid), así que SQLite puede
    // recorrer el índice al revés y cortar en el LIMIT sin tener que
    // materializar+ordenar la tabla entera primero (created_at no tiene
    // índice — con millones de filas, ordenar por esa columna es lo que
    // hacía que esto tardara una eternidad incluso con LIMIT).
    let sql = "SELECT d.slug, d.bundle, d.title, d.source_kind, d.intent_kind, d.tags, d.links, d.created_at, COUNT(c.id)
               FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
               WHERE (?1 IS NULL OR d.bundle = ?1)
               GROUP BY d.id ORDER BY d.id DESC LIMIT ?2 OFFSET ?3";
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![bundle, limit, offset], |r| {
        let tags_json: String = r.get(5)?;
        let links_json: String = r.get(6)?;
        Ok(DocInfo {
            slug: r.get(0)?,
            bundle: r.get(1)?,
            title: r.get(2)?,
            source_kind: r.get(3)?,
            intent_kind: r.get(4)?,
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            links: serde_json::from_str(&links_json).unwrap_or_default(),
            created_at: r.get(7)?,
            chunks: r.get(8)?,
        })
    })?;
    rows.collect()
}

/// Borra un documento — poda también sus vectores del índice usearch.
pub fn delete_document(
    conn: &Connection,
    vec_index: &mut VecIndex,
    slug: &str,
    bundle: &str,
) -> Result<Option<()>, String> {
    let row: Option<(i64, String)> = conn
        .query_row(
            "SELECT id, okf_path FROM documents WHERE slug=?1 AND bundle=?2",
            params![slug, bundle],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    let Some((doc_id, okf_path)) = row else {
        return Ok(None);
    };

    let mut stmt = conn
        .prepare("SELECT id FROM chunks WHERE doc_id=?1")
        .map_err(|e| e.to_string())?;
    let chunk_ids: Vec<i64> = stmt
        .query_map(params![doc_id], |r| r.get(0))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();
    drop(stmt);

    conn.execute("DELETE FROM documents WHERE id=?1", params![doc_id])
        .map_err(|e| e.to_string())?;
    for cid in &chunk_ids {
        vec_index.remove(bundle, *cid);
    }
    let _ = std::fs::remove_file(okf_path);
    Ok(Some(()))
}

/// Vacía TODO el índice derivado (SQLite + vec_index) — usado por
/// `rag_rebuild_index` antes de repoblar desde los .md. Es seguro: el
/// OKF en disco es la fuente de verdad, todo esto es descartable.
pub fn wipe_index(conn: &Connection, vec_index: &mut VecIndex) -> Result<(), String> {
    conn.execute("DELETE FROM chunks", [])
        .map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM documents", [])
        .map_err(|e| e.to_string())?;
    let _ = conn.execute(
        "INSERT INTO chunks_fts(chunks_fts) VALUES ('delete-all')",
        [],
    );
    vec_index.wipe()?;
    Ok(())
}

pub fn document_hash_map(
    conn: &Connection,
    bundle: &str,
) -> rusqlite::Result<FxHashMap<String, String>> {
    let mut stmt = conn.prepare("SELECT slug, content_hash FROM documents WHERE bundle=?1")?;
    let rows = stmt.query_map(params![bundle], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })?;
    rows.collect::<Result<Vec<_>, _>>()
        .map(|v| v.into_iter().collect())
}

#[allow(dead_code)]
/// Reagrupa TODOS los chunks de un documento (ordenados por chunk_index)
/// en un solo texto — se usa para inyectar el documento completo cuando
/// UNO de sus chunks matcheó en la búsqueda, sin tener que leer el .md
/// de disco (evita el I/O y sirve igual si el .md no está o cambió).
pub fn document_full_text(conn: &Connection, doc_id: i64) -> rusqlite::Result<String> {
    let mut stmt =
        conn.prepare("SELECT text FROM chunks WHERE doc_id = ?1 ORDER BY chunk_index ASC")?;
    let parts: Vec<String> = stmt
        .query_map(params![doc_id], |r| r.get::<_, String>(0))?
        .filter_map(Result::ok)
        .collect();
    Ok(parts.join("\n\n"))
}

pub fn list_bundle_dirs(bundle_dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(bundle_dir)
        .map(|it| {
            it.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|name| name.to_str())
                            .map(frontmatter::valid_bundle_name)
                            .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Unión de carpetas físicas en disco + valores distintos de `bundle` en
/// SQLite.
pub fn list_bundles(conn: &Connection, bundle_dir: &Path) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for dir in list_bundle_dirs(bundle_dir) {
        if let Some(name) = dir.file_name().map(|s| s.to_string_lossy().to_string()) {
            set.insert(name);
        }
    }
    if let Ok(mut stmt) = conn.prepare("SELECT DISTINCT bundle FROM documents") {
        if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
            for b in rows.filter_map(Result::ok) {
                set.insert(b);
            }
        }
    }
    set.into_iter().collect()
}

pub fn create_bundle(bundle_dir: &Path, name: &str) -> Result<(), String> {
    if !frontmatter::valid_bundle_name(name) {
        return Err("Nombre de bundle inválido".into());
    }
    let path = bundle_dir.join(name);
    if path.exists() {
        return Err(format!("El bundle '{name}' ya existe."));
    }
    std::fs::create_dir_all(&path).map_err(|e| e.to_string())
}

/// Mueve la carpeta en disco y actualiza `documents.bundle`/`okf_path`.
/// A propósito NO reescribe el campo `resource:`/`title:` dentro de cada
/// .md (drift cosmético, no afecta búsqueda/grafo/reindex).
pub fn rename_bundle(
    conn: &Connection,
    bundle_dir: &Path,
    old: &str,
    new: &str,
) -> Result<usize, String> {
    if !frontmatter::valid_bundle_name(old) || !frontmatter::valid_bundle_name(new) {
        return Err("Nombre de bundle inválido".into());
    }
    let old_path = bundle_dir.join(old);
    let new_path = bundle_dir.join(new);
    if !old_path.exists() {
        return Err(format!("El bundle '{old}' no existe en disco."));
    }
    if new_path.exists() {
        return Err(format!("Ya existe una carpeta '{new}'."));
    }
    std::fs::rename(&old_path, &new_path).map_err(|e| e.to_string())?;
    let n = conn
        .execute(
            "UPDATE documents SET bundle=?1 WHERE bundle=?2",
            params![new, old],
        )
        .map_err(|e| e.to_string())?;
    let old_str = old_path.to_string_lossy().to_string();
    let new_str = new_path.to_string_lossy().to_string();
    let _ = conn.execute(
        "UPDATE documents SET okf_path = REPLACE(okf_path, ?1, ?2) WHERE bundle=?3",
        params![old_str, new_str, new],
    );
    Ok(n)
}

/// Borra todos los documentos de un bundle (podando también sus
/// vectores del índice) y la carpeta entera en disco.
pub fn delete_bundle(
    conn: &Connection,
    vec_index: &mut VecIndex,
    bundle_dir: &Path,
    name: &str,
) -> Result<usize, String> {
    if !frontmatter::valid_bundle_name(name) {
        return Err("Nombre de bundle inválido".into());
    }

    // Mover primero a una carpeta temporal en el mismo volumen es atómico. Si
    // algo falla al actualizar el índice/SQLite se puede restaurar sin haber
    // perdido el contenido fuente.
    let path = bundle_dir.join(name);
    let staged = if path.exists() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let staged = bundle_dir.join(format!(".deleting-{name}-{nonce}"));
        std::fs::rename(&path, &staged).map_err(|e| e.to_string())?;
        Some(staged)
    } else {
        None
    };

    let mut transaction_open = false;
    let result = (|| -> Result<usize, String> {
        conn.execute_batch("BEGIN IMMEDIATE")
            .map_err(|e| e.to_string())?;
        transaction_open = true;
        let n = conn
            .execute("DELETE FROM documents WHERE bundle=?1", params![name])
            .map_err(|e| e.to_string())?;
        vec_index.remove_bundle_dir(name)?;
        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;
        transaction_open = false;
        Ok(n)
    })();

    if let Err(e) = result {
        if transaction_open {
            let _ = conn.execute_batch("ROLLBACK");
        }
        if let Some(staged) = &staged {
            let _ = std::fs::rename(staged, &path);
        }
        return Err(e);
    }

    if let Some(staged) = staged {
        if let Err(e) = std::fs::remove_dir_all(&staged) {
            // SQLite y el índice ya fueron eliminados correctamente. La
            // carpeta temporal queda fuera de la lista de bundles válidos y
            // se registra para limpieza manual en vez de reportar un falso
            // fallo de la operación ya confirmada.
            crate::plugin_log(&format!(
                "WARN bundle '{name}' eliminado, pero no se pudo limpiar {}: {e}",
                staged.display()
            ));
        }
    }

    result
}
