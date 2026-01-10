use std::str;

use crate::BumpArena;

#[derive(Debug, Clone)]
pub struct FastaEntry<'a> {
    pub id: &'a str,
    pub desc: &'a str,
    pub data: &'a str,
}

/// Parses a FASTA byte slice and stores the strings into the Arena.
pub fn parse_fasta_into_arena<'a>(
    arena: &'a BumpArena,
    raw_bytes: &[u8],
) -> Result<Box<[FastaEntry<'a>]>, String> {
    let mut entries = Vec::new();

    let mut current_header: Option<(&'a str, &'a str)> = None;

    // this is unnecessary tbh.
    let mut current_seq_buf: Vec<u8> = Vec::with_capacity(4096);

    let mut ptr = 0;
    while ptr < raw_bytes.len() {
        let end = raw_bytes[ptr..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|i| ptr + i)
            .unwrap_or(raw_bytes.len());

        let mut line_content = &raw_bytes[ptr..end];
        if line_content.ends_with(b"\r") {
            line_content = &line_content[..line_content.len() - 1];
        }

        if line_content.is_empty() {
            ptr = end + 1;
            continue;
        }

        if line_content[0] == b'>' {
            if let Some((stored_id, stored_desc)) = current_header {
                let seq_str = str::from_utf8(&current_seq_buf)
                    .map_err(|_| format!("Invalid UTF-8 in sequence data for ID: {}", stored_id))?;

                let stored_seq = arena.alloc_str(seq_str);

                entries.push(FastaEntry {
                    id: stored_id,
                    desc: stored_desc,
                    data: stored_seq,
                });
            }

            let header_text = &line_content[1..]; // Skip '>'

            let space_pos = header_text.iter().position(|&b| b == b' ');
            let (raw_id, raw_desc) = match space_pos {
                Some(p) => (&header_text[..p], &header_text[p + 1..]),
                None => (header_text, &[] as &[u8]),
            };

            let id_str = str::from_utf8(raw_id)
                .map_err(|_| "Invalid UTF-8 in FASTA Header ID".to_string())?;
            let desc_str = str::from_utf8(raw_desc)
                .map_err(|_| format!("Invalid UTF-8 in FASTA Description for ID: {}", id_str))?;

            let stored_id = arena.alloc_str(id_str);
            let stored_desc = arena.alloc_str(desc_str);

            current_header = Some((stored_id, stored_desc));
            current_seq_buf.clear();
        } else {
            if current_header.is_none() {
                return Err("Parse Error: Found sequence data before the first header (line starting with >)".to_string());
            }

            current_seq_buf.extend_from_slice(line_content);
        }

        ptr = end + 1;
    }

    if let Some((stored_id, stored_desc)) = current_header {
        let seq_str = str::from_utf8(&current_seq_buf)
            .map_err(|_| format!("Invalid UTF-8 in sequence data for ID: {}", stored_id))?;

        let stored_seq = arena.alloc_str(seq_str);

        entries.push(FastaEntry {
            id: stored_id,
            desc: stored_desc,
            data: stored_seq,
        });
    }

    Ok(entries.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_fasta() {
        let arena = BumpArena::new(4096);
        let raw = b">seq1 Human Gene
ATGC
ATGC
>seq2
GGCC";

        let result = parse_fasta_into_arena(&arena, raw);
        assert!(result.is_ok());

        let entries = result.unwrap();
        assert_eq!(entries.len(), 2);

        assert_eq!(entries[0].id, "seq1");
        assert_eq!(entries[0].desc, "Human Gene");
        assert_eq!(entries[0].data, "ATGCATGC");

        assert_eq!(entries[1].id, "seq2");
        assert_eq!(entries[1].desc, "");
        assert_eq!(entries[1].data, "GGCC");
    }

    #[test]
    fn test_error_missing_header() {
        let arena = BumpArena::new(1024);
        let raw = b"ATGC\n>seq1\nATGC"; // Starts with data, not >

        let result = parse_fasta_into_arena(&arena, raw);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Parse Error: Found sequence data before the first header (line starting with >)"
        );
    }

    #[test]
    fn test_error_invalid_utf8() {
        let arena = BumpArena::new(1024);
        let raw = b">seq1\nATG\xFFC";

        let result = parse_fasta_into_arena(&arena, raw);
        assert!(result.is_err());
        println!("result {:?}", result);
        assert!(
            result
                .unwrap_err()
                .contains("Invalid UTF-8 in sequence data for ID: seq1")
        );
    }

    #[test]
    fn test_memory_layout_locality() {
        let arena = BumpArena::new(4096);
        let raw = b">A B
C";
        // This input produces: ID="A", Desc="B", Data="C"
        let entries = parse_fasta_into_arena(&arena, raw).unwrap();
        let e = &entries[0];

        let p_id = e.id.as_ptr() as usize;
        let p_desc = e.desc.as_ptr() as usize;
        let p_data = e.data.as_ptr() as usize;

        // Verify logical ordering in memory: ID < Desc < Data
        assert!(p_id < p_desc);
        assert!(p_desc < p_data);

        // Verify tightness (optional, depends on alignment)
        // "A" is 1 byte + alignment padding
        // "B" is 1 byte + alignment padding
        println!("ID: {:x}, Desc: {:x}, Data: {:x}", p_id, p_desc, p_data);
    }
}

#[cfg(feature = "extensive_tests")]
#[cfg(test)]
mod test_full {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn test_load_all_fasta_files() {
        let arena_size = 1024 * 1024 * 1024;
        let arena = BumpArena::new(arena_size);

        println!("\n--- [Extensive Test] Started ---");
        println!("Arena Capacity: {} MB", arena.capacity() / 1024 / 1024);

        let data_path = Path::new("data/fasta");
        if !data_path.exists() {
            panic!(
                "Directory ./data/fasta/ not found. \
                Please run the python download script first."
            );
        }

        let mut total_entries = 0;
        let mut total_bytes_read = 0;
        let global_start = Instant::now();

        let paths = fs::read_dir(data_path).expect("Could not read data dir");

        for path_result in paths {
            let path = path_result.expect("Error reading path").path();

            if path.is_file() {
                let filename = path.file_name().unwrap().to_string_lossy();
                println!("Loading: {}", filename);

                let raw_bytes = fs::read(&path).expect("Failed to read file bytes");
                total_bytes_read += raw_bytes.len();

                let parse_start = Instant::now();

                match parse_fasta_into_arena(&arena, &raw_bytes) {
                    Ok(entries) => {
                        let duration = parse_start.elapsed();

                        let mut seq_len_sum: usize = 0;
                        for entry in entries.iter() {
                            seq_len_sum += entry.data.len();
                        }

                        println!(
                            "  -> Parsed {} entries in {:.2?}. (Total Seq Len: {})",
                            entries.len(),
                            duration,
                            seq_len_sum
                        );

                        total_entries += entries.len();
                    }
                    Err(e) => {
                        panic!("Failed to parse file {:?}: {}", filename, e);
                    }
                }
            }
        }

        let total_duration = global_start.elapsed();

        println!("\n--- [Extensive Test] Summary ---");
        println!("Total Files Loaded:  {} bytes", total_bytes_read);
        println!("Total Entries:       {}", total_entries);
        println!("Arena Memory Used:   {} MB", arena.used() / 1024 / 1024);
        println!("Total Time:          {:.2?}", total_duration);

        assert!(
            total_entries > 0,
            "No entries parsed! Is the data folder empty?"
        );
    }
}
