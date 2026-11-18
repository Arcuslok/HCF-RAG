//! frontmatter.rs — formato OKF: `---\n<frontmatter>\n---\n\n<cuerpo>`.
//! Port directo de `parse_okf`/`write_okf_file`/`slugify` de plugin.py —
//! a propósito NO es un parser YAML de verdad (no hace falta la
//! dependencia): el frontmatter que escribe este mismo plugin es siempre
//! `clave: valor` simple o `clave: ["a", "b"]`, así que un parser línea
//! por línea alcanza y es lo que ya usa la versión Python — si algún día
//! alguien edita un .md a mano con YAML más complejo, esto lo va a leer
//! de forma degradada (mejor eso que sumar una dependencia grande por un
//! caso que no ocurre en la práctica).

use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::OnceLock;

pub struct ParsedOkf {
    pub title: String,
    pub tags: Vec<String>,
    pub links: Vec<String>,
    pub source_kind: String,
    pub intent_kind: String,
    pub body: String,
    pub hash: String,
}

/// Un bundle es tanto un nombre de directorio como parte de una URI
/// `rag://`. Restringirlo a un único componente evita que una entrada del
/// usuario pueda salir de `bundle_dir` mediante `..` o separadores de ruta.
pub fn valid_bundle_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn content_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn parse_bracket_list(val: &str) -> Vec<String> {
    let inner = &val[1..val.len() - 1];
    if inner.trim().is_empty() {
        return vec![];
    }
    // extrae "..." o '...' — mismo criterio laxo que el regex de Python
    let mut items = vec![];
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' || c == '\'' {
            let quote = c;
            let mut item = String::new();
            for c2 in chars.by_ref() {
                if c2 == quote {
                    break;
                }
                item.push(c2);
            }
            items.push(item);
        }
    }
    items
}

/// Extrae palabras clave de acción del body: comandos tipo `-> <CMD:play>`
/// y palabras a la izquierda de `->` en líneas de mapeo.
fn extract_action_keywords(body: &str) -> Vec<String> {
    static RE_CMD: OnceLock<Regex> = OnceLock::new();
    static RE_ARROW: OnceLock<Regex> = OnceLock::new();

    let re_cmd = RE_CMD.get_or_init(|| Regex::new(r"->\s*<CMD:(\w+)").unwrap());
    let re_arrow = RE_ARROW.get_or_init(|| Regex::new(r"^([a-záéíóúñü\s]+)\s*->").unwrap());

    let mut keywords = FxHashSet::default();
    for line in body.lines() {
        let line = line.trim().to_lowercase();
        // Extrae "play", "pause", "vol"... de -> <CMD:play>
        for cap in re_cmd.captures_iter(&line) {
            if let Some(m) = cap.get(1) {
                keywords.insert(m.as_str().to_string());
            }
        }
        // Extrae "reproduce", "pon", "pausa"... de la parte izquierda de ->
        if let Some(cap) = re_arrow.captures(&line) {
            if let Some(m) = cap.get(1) {
                for word in m.as_str().split_whitespace() {
                    let w = word.trim();
                    if w.len() > 2 && !w.starts_with('<') {
                        keywords.insert(w.to_string());
                    }
                }
            }
        }
    }
    keywords.into_iter().collect()
}

/// Mismo comportamiento que `parse_okf` en Python: si no hay frontmatter
/// bien formado, devuelve el texto entero como cuerpo con los demás
/// campos vacíos (nunca falla — un .md raro no debe tumbar un reindex).
pub fn parse_okf(text: &str) -> ParsedOkf {
    if !text.starts_with("---") {
        return ParsedOkf {
            title: String::new(),
            tags: vec![],
            links: vec![],
            source_kind: String::new(),
            intent_kind: String::new(),
            body: text.to_string(),
            hash: content_hash(text),
        };
    }
    let Some(end) = text[3..].find("\n---").map(|i| i + 3) else {
        return ParsedOkf {
            title: String::new(),
            tags: vec![],
            links: vec![],
            source_kind: String::new(),
            intent_kind: String::new(),
            body: text.to_string(),
            hash: content_hash(text),
        };
    };
    let fm_text = text[3..end].trim();
    let body = text[end + 4..].trim_start_matches('\n').to_string();

    let mut fm: FxHashMap<String, String> = FxHashMap::default();
    let mut lists: FxHashMap<String, Vec<String>> = FxHashMap::default();
    for line in fm_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_string();
        let val = val.trim();
        if val.starts_with('[') && val.ends_with(']') {
            lists.insert(key, parse_bracket_list(val));
        } else if (val.starts_with('"') && val.ends_with('"') && val.len() >= 2)
            || (val.starts_with('\'') && val.ends_with('\'') && val.len() >= 2)
        {
            fm.insert(key, val[1..val.len() - 1].to_string());
        } else {
            fm.insert(key, val.to_string());
        }
    }

    let mut auto_tags = extract_action_keywords(&body);
    if let Some(existing_tags) = lists.get("tags") {
        auto_tags.extend(existing_tags.iter().cloned());
    }

    ParsedOkf {
        title: fm.get("title").cloned().unwrap_or_default(),
        tags: auto_tags.into_iter().collect(),
        links: lists.get("links").cloned().unwrap_or_default(),
        source_kind: fm
            .get("source_kind")
            .cloned()
            .unwrap_or_else(|| "okf".to_string()),
        intent_kind: fm.get("intent_kind").cloned().unwrap_or_default(),
        body,
        hash: content_hash(text),
    }
}

/// Separa el heading `# título` que `write_okf` antepone al cuerpo, para
/// mostrar en un editor solo el texto real (no el markdown completo del
/// archivo). Si el body no matchea ese patrón exacto (p. ej. alguien
/// editó el .md a mano y le sacó el heading, o le puso otro título),
/// devuelve el body entero sin tocar nada — degradado, no roto.
pub fn split_heading(body: &str, title: &str) -> String {
    let heading = format!("# {title}");
    match body.strip_prefix(&heading) {
        Some(rest) => rest.trim_start_matches('\n').to_string(),
        None => body.to_string(),
    }
}

fn yaml_str(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', " ")
    )
}
fn yaml_list(items: &[String]) -> String {
    if items.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            items
                .iter()
                .map(|s| yaml_str(s))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Escribe el .md y devuelve su `content_hash` — mismo formato exacto que
/// `write_okf_file` en Python (mismo orden de campos), para que un
/// archivo escrito por un lado se parsee igual del otro.
pub fn write_okf(
    bundle_dir: &Path,
    bundle: &str,
    slug: &str,
    title: &str,
    text: &str,
    source_kind: &str,
    intent_kind: &str,
    tags: &[String],
    links: &[String],
    timestamp: &str,
) -> std::io::Result<(std::path::PathBuf, String)> {
    if !valid_bundle_name(bundle) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Nombre de bundle inválido",
        ));
    }
    let desc: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(180)
        .collect();
    let dir = bundle_dir.join(bundle);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{slug}.md"));
    // `intent_kind` solo se escribe si viene seteado — un doc sin
    // clasificar no necesita la línea (parse_okf ya devuelve "" por
    // default si no la encuentra), así los .md viejos/sin clasificar
    // quedan igual que antes de este campo.
    let intent_line = if intent_kind.is_empty() {
        String::new()
    } else {
        format!("intent_kind: {intent_kind}\n")
    };
    let content = format!(
        "---\ntype: rag-document\ntitle: {}\ndescription: {}\nresource: rag://{bundle}/{slug}\ntags: {}\nlinks: {}\ntimestamp: {timestamp}\nsource_kind: {source_kind}\n{intent_line}---\n\n# {title}\n\n{text}\n",
        yaml_str(title), yaml_str(&desc), yaml_list(tags), yaml_list(links)
    );
    std::fs::write(&path, &content)?;
    Ok((path, content_hash(&content)))
}

/// NFKD + sacar combining marks + todo lo no alfanumérico a guiones —
/// mismo criterio que `slugify` en Python (así los slugs coinciden si
/// algún día se comparan entre las dos implementaciones).
pub fn slugify(title: &str) -> String {
    let lower = title.trim().to_lowercase();
    let decomposed: String = unicode_nfkd(&lower);
    let mut out = String::new();
    let mut last_dash = false;
    for c in decomposed.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        format!(
            "doc-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        )
    } else {
        out
    }
}

/// Descomposición NFKD mínima manual para las letras acentuadas latinas
/// más comunes (á,é,í,ó,ú,ñ,ü + mayúsculas) — cubre español, que es lo
/// que importa acá. No es una implementación Unicode completa (para eso
/// haría falta la crate `unicode-normalization`), pero el resultado final
/// pasa igual por el filtro `is_ascii_alphanumeric`, así que cualquier
/// acento no cubierto simplemente se convierte en un guion en vez de
/// preservar la letra base — degradado, no roto.
fn unicode_nfkd(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!("ame_fm_test_{}", std::process::id()));
        let (path, hash1) = write_okf(
            &dir,
            "wikipedia",
            "utulau",
            "'Utulau",
            "'Utulau es una localidad del distrito de Nukunuku, en Tonga.",
            "wikipedia-zim",
            "info",
            &["geografia".into(), "tonga".into()],
            &["nukunuku".into()],
            "2026-07-26T00:00:00Z",
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed = parse_okf(&text);
        assert_eq!(parsed.title, "'Utulau");
        assert_eq!(parsed.tags, vec!["geografia", "tonga"]);
        assert_eq!(parsed.links, vec!["nukunuku"]);
        assert_eq!(parsed.source_kind, "wikipedia-zim");
        assert_eq!(parsed.intent_kind, "info");
        assert!(parsed.body.contains("Nukunuku"));
        assert_eq!(parsed.hash, hash1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn slugify_matches_python_semantics() {
        assert_eq!(slugify("'74 Jailbreak"), "74-jailbreak");
        assert_eq!(slugify("Álgebra lineal"), "algebra-lineal");
        assert_eq!(slugify("  Hola, Mundo!  "), "hola-mundo");
    }

    #[test]
    fn bundle_names_are_single_safe_path_components() {
        for name in ["default", "mi_bundle-2", "ABC_123"] {
            assert!(valid_bundle_name(name), "{name} debería ser válido");
        }
        for name in ["", ".", "..", "../otro", "a/b", "a\\b", "con espacio"] {
            assert!(!valid_bundle_name(name), "{name} no debería ser válido");
        }
    }
}
