use std::collections::HashMap;

use db::{
    Column, DbBuilder, Dna2NaiveWildcard, Dna2TableBuilder, FullScan, LikeCompileOptions,
    LikePattern, QueryScratch, RowId, SortedRowsProbe, Utf8Kmp, Utf8TableBuilder, execute_like,
};

/// Toy trigram index for demonstrating candidate generation.
///
/// `execute_like` still verifies every candidate row with the compiled LIKE
/// pattern, so false positives from this index remain safe.
#[derive(Debug, Default)]
struct TrigramIndex {
    postings: HashMap<[u8; 3], Vec<RowId>>,
}

impl TrigramIndex {
    fn build<C>(column: &C) -> Self
    where
        C: Column<Symbol = u8>,
    {
        let mut postings: HashMap<[u8; 3], Vec<RowId>> = HashMap::new();

        for row in 0..column.row_count() {
            let mut window = [0u8; 3];
            let mut filled = 0usize;

            for sym in column.symbols(row) {
                if filled < 3 {
                    window[filled] = sym;
                    filled += 1;
                } else {
                    window = [window[1], window[2], sym];
                }

                if filled == 3 {
                    postings.entry(window).or_default().push(row);
                }
            }
        }

        for rows in postings.values_mut() {
            rows.sort_unstable();
            rows.dedup();
        }

        Self { postings }
    }

    fn postings(&self, gram: [u8; 3]) -> &[RowId] {
        self.postings.get(&gram).map(Vec::as_slice).unwrap_or(&[])
    }
}

fn first_trigram(bytes: &[u8]) -> Option<[u8; 3]> {
    let w = bytes.windows(3).next()?;
    Some([w[0], w[1], w[2]])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------------------------------------------------------
    // UTF-8 byte LIKE with KMP literal search.
    // KMP does not support `_`, so `_` is compiled to Skip(1). KMP prefix
    // tables are built once inside LikePattern and reused for every row.
    // ---------------------------------------------------------------------
    let mut docs = Utf8TableBuilder::new("docs");
    docs.push_str("hello ACG world");
    docs.push_str("no match here");
    docs.push_str("ACGT prefix");
    docs.push_str("almost AGG");

    let mut dbb = DbBuilder::new();
    let docs_id = dbb.add_utf8_table(docs)?;
    let db = dbb.freeze();
    let docs = db.utf8_table(docs_id)?;
    let text = docs.text();

    let pattern = LikePattern::<Utf8Kmp>::compile("%ACG%")?;

    let mut scan = FullScan::new(text.row_count(), 1024);
    let mut scratch = QueryScratch::default();
    let mut matches = Vec::<RowId>::new();

    let stats = execute_like(&text, &mut scan, &pattern, &mut scratch, &mut matches);
    println!("full scan matches: {matches:?}, stats: {stats:?}");

    // Same verifier, but candidates come from a toy trigram index.
    let trigram = TrigramIndex::build(&text);
    if let Some(gram) = pattern.longest_indexable_literal().and_then(first_trigram) {
        let rows = trigram.postings(gram);
        let mut probe = SortedRowsProbe::new(rows, 1024);
        let mut indexed_matches = Vec::<RowId>::new();
        let stats = execute_like(
            &text,
            &mut probe,
            &pattern,
            &mut scratch,
            &mut indexed_matches,
        );
        println!("trigram matches: {indexed_matches:?}, stats: {stats:?}");
    }

    // ---------------------------------------------------------------------
    // DNA2 LIKE with an algorithm that accepts `_` inside literal fragments.
    // Pattern A_G% is compiled as Literal("A_G"), Any. The row verifier reads
    // packed DNA2 rows directly; no UTF-8 string materialization is required.
    // ---------------------------------------------------------------------
    let mut reads = Dna2TableBuilder::new("reads");
    reads.push_str("ACGT")?;
    reads.push_str("AGGT")?;
    reads.push_str("TTTT")?;

    let mut dbb = DbBuilder::new();
    let reads_id = dbb.add_dna2_table(reads)?;
    let db = dbb.freeze();
    let reads = db.dna2_table(reads_id)?;
    let seq = reads.sequence();

    let dna_pattern = LikePattern::<Dna2NaiveWildcard>::compile("A_G%")?;
    let mut scan = FullScan::new(seq.row_count(), 1024);
    let mut scratch = QueryScratch::default();
    let mut dna_matches = Vec::<RowId>::new();

    let stats = execute_like(
        &seq,
        &mut scan,
        &dna_pattern,
        &mut scratch,
        &mut dna_matches,
    );
    println!("dna direct-wildcard matches: {dna_matches:?}, stats: {stats:?}");

    // If you want `_` to become Skip(1) even for a wildcard-capable algorithm,
    // force this at compile time. This may expose smaller exact fragments to
    // indexes, at the cost of not using the algorithm's direct wildcard path.
    let dna_skip_pattern = LikePattern::<Dna2NaiveWildcard>::compile_with_options(
        "A_G%",
        LikeCompileOptions {
            pass_underscore_to_algorithm: false,
        },
    )?;
    assert!(dna_skip_pattern.has_skip());

    Ok(())
}
