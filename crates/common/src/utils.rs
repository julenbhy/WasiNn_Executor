use std::io::Cursor;
use base64::{engine::general_purpose, Engine as _};

#[inline(always)]
pub fn b64_decode(b64_string: String) -> anyhow::Result<Vec<u8>> {
    let time = std::time::Instant::now();
    let module_bytes: Vec<u8> = general_purpose::STANDARD.decode(b64_string)?;
    println!("base64 decoding took {} ms", time.elapsed().as_millis());
    Ok(module_bytes)
}

pub fn unzip(bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    let mut target = Cursor::new(Vec::with_capacity(bytes.len()));
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).unwrap();

    let mut file = archive.by_index(0)?;

    std::io::copy(&mut file, &mut target)?;

    Ok(target.into_inner())
}



use serde_json::Value;
pub fn print_json_structure(value: &Value, max_preview_len: usize) {
    fn helper(value: &Value, max_preview_len: usize, level: usize) {
        let indent = "  ".repeat(level);

        match value {
            Value::Object(map) => {
                println!("{}{{", indent);
                for (key, val) in map {
                    print!("{}  \"{}\": ", indent, key);
                    match val {
                        Value::Object(_) | Value::Array(_) => {
                            println!();
                            helper(val, max_preview_len, level + 1);
                        }
                        _ => {
                            let preview = format!("{}", val);
                            let shortened = if preview.len() > max_preview_len {
                                format!("{}...", &preview[..max_preview_len])
                            } else {
                                preview
                            };
                            println!("{}", shortened);
                        }
                    }
                }
                println!("{}}}", indent);
            }

            Value::Array(arr) => {
                println!("{}[", indent);
                if let Some(first) = arr.first() {
                    helper(first, max_preview_len, level + 1);
                } else {
                    println!("{}  (empty array)", indent);
                }
                println!("{}]", indent);
            }

            _ => {
                let preview = format!("{}", value);
                let shortened = if preview.len() > max_preview_len {
                    format!("{}...", &preview[..max_preview_len])
                } else {
                    preview
                };
                println!("{}{}", indent, shortened);
            }
        }
    }

    helper(value, max_preview_len, 0);
}