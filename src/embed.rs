//! embed.rs — chunking + "hashing vectorizer" (mismo esquema que la
//! versión Python: palabras + n-gramas de 3 caracteres, hasheados con
//! CRC32 a un vector de 512 dims, L2-normalizado). Se mantiene el mismo
//! esquema para que embeddings viejos sigan siendo comparables.

use std::sync::OnceLock;

pub const EMBED_DIM: usize = 512;
pub const CHUNK_MAX: usize = 900;
pub const CHUNK_OVERLAP: usize = 150;

fn crc32_table() -> &'static [u32; 256] {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();

    TABLE.get_or_init(|| {
        let mut table = [0u32; 256];

        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            table[i as usize] = c;
        }

        table
    })
}

/// CRC32 compatible con zlib.crc32.
fn crc32(bytes: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc = 0xFFFF_FFFFu32;

    for &b in bytes {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }

    !crc
}

/// Hash de trigrama con prefijo "g:" sin asignar Strings.
/// Máximo posible: 2 bytes de prefijo + 3 chars UTF-8 (hasta 4 bytes cada uno) = 14 bytes.
fn crc32_trigram(a: char, b: char, c: char) -> u32 {
    let mut buf = [0u8; 16];
    buf[0] = b'g';
    buf[1] = b':';

    let mut len = 2usize;
    let mut tmp = [0u8; 4];

    for ch in [a, b, c] {
        let s = ch.encode_utf8(&mut tmp);
        let bytes = s.as_bytes();
        buf[len..len + bytes.len()].copy_from_slice(bytes);
        len += bytes.len();
    }

    crc32(&buf[..len])
}

pub fn embed(text: &str) -> Vec<f32> {
    let mut vec = vec![0f32; EMBED_DIM];

    // to_lowercase asigna una vez; después trabajamos sobre slices.
    let lower = text.to_lowercase();

    for tok in lower.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if tok.is_empty() {
            continue;
        }

        // Palabra completa.
        let idx = (crc32(tok.as_bytes()) as usize) % EMBED_DIM;
        vec[idx] += 1.0;

        // Trigramas de caracteres sin coleccionar el token entero.
        let mut ring = ['\u{0}'; 3];
        let mut count = 0usize;

        for ch in tok.chars() {
            if count < 3 {
                ring[count] = ch;
                count += 1;
            } else {
                ring[0] = ring[1];
                ring[1] = ring[2];
                ring[2] = ch;
            }

            if count >= 3 {
                let idx = (crc32_trigram(ring[0], ring[1], ring[2]) as usize) % EMBED_DIM;
                vec[idx] += 0.4;
            }
        }
    }

    // Normalización L2.
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }

    vec
}

/// Igual estrategia que `chunk_text` en Python: agrupa párrafos hasta
/// CHUNK_MAX caracteres, y corta con solape los párrafos que por sí solos
/// ya superan el máximo.
pub fn chunk_text(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let paras: Vec<&str> = text.split("\n\n").collect();
    let mut chunks = Vec::new();

    let mut buf = String::new();
    let mut buf_chars = 0usize;

    for raw_p in paras {
        let p = raw_p.trim();
        if p.is_empty() {
            continue;
        }

        let p_chars = p.chars().count();

        // Entra en el buffer actual.
        if buf_chars + p_chars + 1 <= CHUNK_MAX {
            if buf.is_empty() {
                buf.push_str(p);
                buf_chars = p_chars;
            } else {
                buf.push('\n');
                buf.push_str(p);
                buf_chars += 1 + p_chars;
            }
            continue;
        }

        // Flush del buffer actual.
        if !buf.is_empty() {
            chunks.push(std::mem::take(&mut buf));
            buf_chars = 0;
        }

        // El párrafo entra solo.
        if p_chars <= CHUNK_MAX {
            buf.push_str(p);
            buf_chars = p_chars;
        } else {
            // Párrafo gigante: cortar por caracteres con overlap.
            let chars: Vec<char> = p.chars().collect();
            let mut start = 0usize;

            while start < chars.len() {
                let end = (start + CHUNK_MAX).min(chars.len());
                chunks.push(chars[start..end].iter().collect());

                if end == chars.len() {
                    break;
                }

                start = end.saturating_sub(CHUNK_OVERLAP);
            }
        }
    }

    if !buf.is_empty() {
        chunks.push(buf);
    }

    chunks
}
