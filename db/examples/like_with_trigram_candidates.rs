//! Example: `TrigramIndex` provides candidates, and the compiled LIKE verifier
//! remains the correctness gate.

use db::{
    Column, DbBuilder, FullScan, LikePattern, QueryScratch, RowId, TrigramIndex, Utf8TableBuilder,
    execute_like,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dbb = DbBuilder::new();
    let mut docs = Utf8TableBuilder::new("docs");
    docs.push_str("abc hello z");
    docs.push_str("abc nope");
    docs.push_str("something hello");
    docs.push_str("hell but not hello");
    let docs_id = dbb.add_utf8_table(docs)?;
    let db = dbb.freeze();

    let col = db.utf8_table(docs_id)?.text();
    let idx = TrigramIndex::build(&col);

    let like = LikePattern::<db::Utf8Kmp>::compile("%hello%")?;
    let mut probe = idx
        .probe_longest_like_literal(&like, 4096)
        .expect("%hello% has an indexable literal of length >= 3");

    let mut scratch = QueryScratch::default();
    let mut rows = Vec::<RowId>::new();
    let stats = execute_like(&col, &mut probe, &like, &mut scratch, &mut rows);

    // A full scan with the same verifier would produce the same rows.
    let mut scan = FullScan::new(col.row_count(), 4096);
    let mut scan_rows = Vec::<RowId>::new();
    execute_like(&col, &mut scan, &like, &mut scratch, &mut scan_rows);

    println!("indexed rows = {rows:?}");
    println!("full scan rows = {scan_rows:?}");
    println!("stats = {stats:?}");
    Ok(())
}
