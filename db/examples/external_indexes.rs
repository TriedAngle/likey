//! External index example using the candidate-provider/verifier/sink API.
//!
//! The toy trigram index here supplies candidate rows. `AcceptAll` stands in for
//! a real compiled verifier so the example can focus on the query execution
//! seam.

use std::collections::HashMap;

use db::{
    AcceptAll, Column, DbBuilder, Dna2TableBuilder, DnaBase, LenConstraint, QueryScratch, RowId,
    SortedRowsProbe, Utf8TableBuilder, execute_like,
};

/// A toy trigram index usable on both Utf8Column and Dna2Column because both
/// expose `Column<Symbol = u8>`.
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
            let mut w = [0u8; 3];
            let mut filled = 0usize;

            for sym in column.symbols(row) {
                if filled < 3 {
                    w[filled] = sym;
                    filled += 1;
                } else {
                    w = [w[1], w[2], sym];
                }

                if filled == 3 {
                    postings.entry(w).or_default().push(row);
                }
            }
        }

        for rows in postings.values_mut() {
            rows.sort_unstable();
            rows.dedup();
        }

        Self { postings }
    }

    fn probe(&self, gram: [u8; 3]) -> SortedRowsProbe<'_> {
        let rows = self.postings.get(&gram).map(Vec::as_slice).unwrap_or(&[]);
        SortedRowsProbe::new(rows, 4096)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dbb = DbBuilder::new();

    let mut docs = Utf8TableBuilder::new("docs_utf8");
    docs.push_str("ACGT one");
    docs.push_str("TTT nope");
    docs.push_str("xx ACG yy");
    let docs_id = dbb.add_utf8_table(docs)?;

    let mut dna = Dna2TableBuilder::new("reads_dna2");
    dna.push_str("ACGTACGT")?;
    dna.push_str("TTTTTTTT")?;
    dna.push_str("GGGACGTT")?;
    let dna_id = dbb.add_dna2_table(dna)?;

    let db = dbb.freeze();

    // UTF-8 table: trigram is literal bytes.
    let docs_table = db.utf8_table(docs_id)?;
    let docs_col = docs_table.text();
    let docs_index = TrigramIndex::build(&docs_col);
    let mut docs_probe = docs_index.probe(*b"ACG");

    // Placeholder verifier. In your real code this would be the compiled LIKE
    // verifier. Because the index has already filtered by one trigram, AcceptAll
    // returns the rows containing that gram.
    let verifier = AcceptAll::new(LenConstraint::at_least(3));
    let mut scratch = QueryScratch::default();
    let mut docs_matches = Vec::<RowId>::new();
    let docs_stats = execute_like(
        &docs_col,
        &mut docs_probe,
        &verifier,
        &mut scratch,
        &mut docs_matches,
    );

    println!(
        "UTF8 candidate/matches: {:?}, stats: {:?}",
        docs_matches, docs_stats
    );

    // DNA2 table: trigram is logical base codes, not ASCII bytes.
    let dna_table = db.dna2_table(dna_id)?;
    let dna_col = dna_table.sequence();
    let dna_index = TrigramIndex::build(&dna_col);
    let acg = [DnaBase::A.code(), DnaBase::C.code(), DnaBase::G.code()];
    let mut dna_probe = dna_index.probe(acg);

    let mut dna_matches = Vec::<RowId>::new();
    let dna_stats = execute_like(
        &dna_col,
        &mut dna_probe,
        &verifier,
        &mut scratch,
        &mut dna_matches,
    );

    println!(
        "DNA2 candidate/matches: {:?}, stats: {:?}",
        dna_matches, dna_stats
    );

    Ok(())
}
