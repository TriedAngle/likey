//! Minimal full-scan query over both storage types.

use db::{
    AcceptAll, Column, DbBuilder, Dna2TableBuilder, FullScan, LenConstraint, QueryScratch, RowId,
    Utf8TableBuilder, execute_like,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dbb = DbBuilder::new();

    let mut docs = Utf8TableBuilder::new("docs");
    docs.push_str("a");
    docs.push_str("abcd");
    docs.push_str("abcdef");
    let docs_id = dbb.add_utf8_table(docs)?;

    let mut dna = Dna2TableBuilder::new("dna");
    dna.push_str("AC")?;
    dna.push_str("ACGT")?;
    dna.push_str("ACGTAC")?;
    let dna_id = dbb.add_dna2_table(dna)?;

    let db = dbb.freeze();
    let verifier = AcceptAll::new(LenConstraint::between(2, 4));
    let mut scratch = QueryScratch::default();

    let docs_col = db.utf8_table(docs_id)?.text();
    let mut docs_scan = FullScan::new(docs_col.row_count(), 1024);
    let mut docs_rows = Vec::<RowId>::new();
    execute_like(
        &docs_col,
        &mut docs_scan,
        &verifier,
        &mut scratch,
        &mut docs_rows,
    );
    println!("docs rows with length 2..=4: {:?}", docs_rows);

    let dna_col = db.dna2_table(dna_id)?.sequence();
    let mut dna_scan = FullScan::new(dna_col.row_count(), 1024);
    let mut dna_rows = Vec::<RowId>::new();
    execute_like(
        &dna_col,
        &mut dna_scan,
        &verifier,
        &mut scratch,
        &mut dna_rows,
    );
    println!("dna rows with length 2..=4: {:?}", dna_rows);

    Ok(())
}
