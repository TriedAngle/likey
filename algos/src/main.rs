use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant}; // Added Duration and Instant

use algos::{BM, KMP, KmerConfig, KmerSearch, Naive, NaiveScalar, NaiveVectorized, StringSearch};
use clap::Parser;

#[derive(Debug, Clone, clap::ValueEnum)]
enum Algorithm {
    Naive,
    NaiveScalar,
    NaiveVectorized,
    Kmp,
    Bm,
    Kmer,
}

/// Example:
/// time cargo run --release -- -t data/chimpansee_dna.txt -t data/keratin_homo_sapiens.txt -t data/ipsum.txt --pattern "TCGC" -a naive --measure-time
/// time cargo run --release -- -t data/chimpansee_dna.txt --pattern "TCGC" -a kmer --kmer-k 2 --kmer-min-hits 3
/// for wildcard sql like pattern matching (% and _) add --like
#[derive(Debug, clap::Parser)]
#[command(
    name = "string-search",
    about = "Run different string search algorithms on one pattern and one or more texts"
)]
struct Cli {
    #[arg(short, long, value_enum)]
    algo: Algorithm,

    #[arg(short = 't', long = "text", value_name = "TEXT", required = true)]
    texts: Vec<PathBuf>,

    #[arg(long)]
    like: bool,

    #[arg(
        long,
        conflicts_with = "pattern_file",
        required_unless_present = "pattern_file"
    )]
    pattern: Option<String>,

    #[arg(
        long = "pattern-file",
        value_name = "PATTERN_FILE",
        conflicts_with = "pattern",
        required_unless_present = "pattern"
    )]
    pattern_file: Option<PathBuf>,

    #[arg(short = 'e', long = "encoding", default_value = "utf8")]
    encoding: String,

    /// Alphabet size. Either a single number (applied to all inputs)
    /// or one number for each text file plus one for the pattern.
    #[arg(long = "alphabet-size", value_name = "N")]
    alphabet_sizes: Vec<usize>,

    /// K for the K-mer index (only used with --algo kmer)
    #[arg(long = "kmer-k", default_value_t = 8)]
    kmer_k: usize,

    /// Minimum number of k-mer hits on the same diagonal to report a match (only used with --algo kmer)
    #[arg(long = "kmer-min-hits", default_value_t = 3)]
    kmer_min_hits: usize,

    /// Optional output file; if omitted, results are written to stdout
    #[arg(short = 'o', long = "output", value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Measure and print execution time for the search algorithm
    #[arg(long)]
    measure_time: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let encoding = cli.encoding.to_lowercase();
    if encoding != "utf8" && encoding != "utf-8" {
        return Err(format!(
            "Only UTF-8 encoding is supported at the moment (got {})",
            cli.encoding
        )
        .into());
    }

    let pattern = load_pattern(&cli)?;
    if pattern.is_empty() {
        return Err("Pattern must not be empty".into());
    }

    let per_text_and_pattern_alpha = resolve_alphabet_sizes(cli.texts.len(), &cli.alphabet_sizes)?;

    let mut out: Box<dyn Write> = match cli.output {
        Some(ref path) => Box::new(File::create(path)?),
        None => Box::new(io::stdout()),
    };

    writeln!(
        out,
        "# algorithm={:?}, encoding={}, pattern-length={}",
        cli.algo,
        encoding,
        pattern.len()
    )?;

    for (idx, text_path) in cli.texts.iter().enumerate() {
        let text = load_text(text_path)?;

        let (alpha_text, alpha_pattern) = per_text_and_pattern_alpha
            .as_ref()
            .map(|v| (v[idx], v[v.len() - 1]))
            .unwrap_or((None, None));

        if alpha_text.is_some() || alpha_pattern.is_some() {
            writeln!(
                out,
                "# alphabet-size text={:?} pattern={:?} (for text {:?})",
                alpha_text, alpha_pattern, text_path
            )?;
        }

        let (matches, duration) = run_algorithm(&cli, &text, &pattern)?;
        
        writeln!(out, "text={:?}", text_path)?;
        
        if let Some(d) = duration {
            writeln!(out, "execution_time: {}ns", d.as_nanos())?;
        }
        
        writeln!(out, "matches: {:?}", matches)?;
        writeln!(out)?;
    }

    Ok(())
}

fn load_pattern(cli: &Cli) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(ref pat) = cli.pattern {
        Ok(pat.clone())
    } else if let Some(ref path) = cli.pattern_file {
        load_text(path)
    } else {
        Err("Either --pattern or --pattern-file must be provided".into())
    }
}

fn load_text(path: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    if path.as_os_str() == "-" {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    } else {
        let mut file = File::open(path)?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;
        Ok(buf)
    }
}

fn resolve_alphabet_sizes(
    text_count: usize,
    sizes: &[usize],
) -> Result<Option<Vec<Option<usize>>>, Box<dyn std::error::Error>> {
    if sizes.is_empty() {
        return Ok(None);
    }

    if sizes.len() == 1 {
        let mut v = Vec::with_capacity(text_count + 1);
        for _ in 0..=text_count {
            v.push(Some(sizes[0]));
        }
        return Ok(Some(v));
    }

    if sizes.len() == text_count + 1 {
        let v = sizes.iter().copied().map(Some).collect();
        return Ok(Some(v));
    }

    Err(format!(
        "alphabet-size: expected either 1 value or {} values (one per text plus one for the pattern), got {}",
        text_count + 1,
        sizes.len()
    )
    .into())
}

fn run_algorithm(
    cli: &Cli,
    text: &str,
    pattern: &str,
) -> Result<(Vec<usize>, Option<Duration>), Box<dyn std::error::Error>> {
    let start = if cli.measure_time {
        Some(Instant::now())
    } else {
        None
    };

    let result = match cli.algo {
        Algorithm::Naive => Naive::find_all((), text, pattern),
        Algorithm::NaiveScalar => NaiveScalar::find_all((), text, pattern),
        Algorithm::NaiveVectorized => NaiveVectorized::find_all((), text, pattern),
        Algorithm::Kmp => KMP::find_all((), text, pattern),
        Algorithm::Bm => BM::find_all((), text, pattern),
        Algorithm::Kmer => {
            let cfg = KmerConfig {
                pattern: pattern.as_bytes().to_vec(),
                k: cli.kmer_k,
                min_hits: cli.kmer_min_hits,
            };
            let index = KmerSearch::build(cfg);
            <KmerSearch as StringSearch>::find_all(index, text, pattern)
        }
    };

    let duration = start.map(|s| s.elapsed());

    Ok((result, duration))
}
