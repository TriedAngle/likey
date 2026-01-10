use std::fs;
use std::path::Path;
use std::time::Instant;

use storage::BumpArena;
use storage::fasta::parse_fasta_into_arena;

pub fn main() {
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
