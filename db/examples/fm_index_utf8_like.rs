use db::{
    DbBuilder, FmIndex, LikePattern, QueryScratch, RowId, Utf8Kmp, Utf8TableBuilder, execute_like,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut docs = Utf8TableBuilder::new("docs");
    docs.push_str("banana");
    docs.push_str("bandana");
    docs.push_str("apple");
    docs.push_str("canary");
    docs.push_str("nothing");

    let mut dbb = DbBuilder::new();
    let docs_id = dbb.add_utf8_table(docs)?;
    let db = dbb.freeze();

    let table = db.utf8_table(docs_id)?;
    let text = table.text();

    // KMP compiles the literal fragment and builds its prefix table once.
    let like = LikePattern::<Utf8Kmp>::compile("%ana%")?;

    // The FM-index is only a candidate producer. `execute_like` still verifies
    // the full LIKE pattern row-by-row.
    let fm = FmIndex::build(&text)?;
    let mut candidates = fm
        .probe_longest_like_literal(&like, 1024)
        .expect("%ana% has one exact literal fragment");

    let mut scratch = QueryScratch::default();
    let mut matches = Vec::<RowId>::new();
    let stats = execute_like(&text, &mut candidates, &like, &mut scratch, &mut matches);

    println!("matching row ids: {matches:?}");
    for row in &matches {
        println!("row {row}: {}", text.row_str(*row)?);
    }
    println!("stats: {stats:?}");

    Ok(())
}
