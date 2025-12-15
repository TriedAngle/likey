use std::path::Path;
use std::process::Command;

// Configuration
const BINARY_NAME: &str = "algos";
const TEXT_FILES: &[&str] = &[
    "data/chimpansee_dna.txt",
    "data/keratin_homo_sapiens.txt",
    "data/ipsum.txt",
];

const PATTERNS: &[(&str, &str)] = &[
    ("TCGC", "Short DNA"),
    ("GATTACA", "Medium DNA"),
    ("Lorem", "Common Word"),
    ("XYZXYZMISSING", "Not Present"),
];

const ALGORITHMS: &[&str] = &[
    "naive",
    "naive-scalar",
    "naive-vectorized",
    "kmp",
    "bm",
    "kmer",
];

#[derive(Debug)]
struct ResultEntry {
    algo: String,
    pattern: String,
    file: String,
    duration_ns: u128,
}

fn main() {
    println!("--- Starting Benchmark Script ---");

    println!("> Building project in release mode...");
    let build_status = Command::new("cargo")
        .args(&["build", "--release"])
        .status()
        .expect("Failed to execute cargo build");

    if !build_status.success() {
        eprintln!("Error: Cargo build failed.");
        std::process::exit(1);
    }

    let binary_path = Path::new("target").join("release").join(BINARY_NAME);
    if !binary_path.exists() {
        eprintln!("Error: Binary not found at {:?}. Check crate name.", binary_path);
        std::process::exit(1);
    }

    let mut results: Vec<ResultEntry> = Vec::new();

    for (pattern, pat_desc) in PATTERNS {
        for algo in ALGORITHMS {
            println!("> Running {} on pattern '{}' ({})", algo, pattern, pat_desc);

            let mut args = vec![
                "--measure-time".to_string(),
                "--pattern".to_string(),
                pattern.to_string(),
                "--algo".to_string(),
                algo.to_string(),
            ];

            for txt in TEXT_FILES {
                args.push("-t".to_string());
                args.push(txt.to_string());
            }

            // Specific tweak for Kmer: k must be <= pattern length
            if *algo == "kmer" {
                // Use k=4 or pattern length if smaller
                let k = std::cmp::min(pattern.len(), 4);
                args.push(format!("--kmer-k"));
                args.push(k.to_string());
                args.push("--kmer-min-hits".to_string());
                // Lower hits for short patterns
                args.push("1".to_string()); 
            }

            let output = Command::new(&binary_path)
                .args(&args)
                .output()
                .expect("Failed to run binary");

            if !output.status.success() {
                eprintln!("  ! Algorithm {} failed on pattern {}", algo, pattern);
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("  ! Error: {}", stderr);
                continue;
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            let parsed_results = parse_output(&stdout, algo, pattern);
            results.extend(parsed_results);
        }
    }

    print_summary_table(&results);
}

fn parse_output(output: &str, algo: &str, pattern: &str) -> Vec<ResultEntry> {
    let mut entries = Vec::new();
    let mut current_file = String::new();

    for line in output.lines() {
        let line = line.trim();
        
        if line.starts_with("text=") {
            current_file = line
                .trim_start_matches("text=\"")
                .trim_end_matches('"')
                .to_string();
        }
        
        if line.starts_with("execution_time:") {
            if let Some(ns_str) = line.split_whitespace().nth(1) {
                let ns_val = ns_str.trim_end_matches("ns");
                if let Ok(ns) = ns_val.parse::<u128>() {
                    entries.push(ResultEntry {
                        algo: algo.to_string(),
                        pattern: pattern.to_string(),
                        file: current_file.clone(),
                        duration_ns: ns,
                    });
                }
            }
        }
    }
    entries
}

fn print_summary_table(results: &[ResultEntry]) {
    println!("\n\n{:=^80}", " RESULTS SUMMARY ");
    println!(
        "{:<18} | {:<15} | {:<25} | {:>15}",
        "Algorithm", "Pattern", "File", "Time (µs)"
    );
    println!("{:-^80}", "");

    for entry in results {
        let micros = entry.duration_ns as f64 / 1000.0;
        
        let short_file = Path::new(&entry.file)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();

        println!(
            "{:<18} | {:<15} | {:<25} | {:>15.2}",
            entry.algo, 
            entry.pattern.chars().take(12).collect::<String>(), 
            short_file, 
            micros
        );
    }
    println!("{:=^80}", " END ");
}
