use db::{
    BM, Column, DbBuilder, FftStr1, FullScan, LikePattern, NaiveMixed, QueryScratch, RowId,
    StdSearch, TwoWay2, Utf8TableBuilder, execute_like,
};

fn run<A>(name: &str, pattern: &str, text: db::Utf8Column<'_>)
where
    A: db::LiteralAlgorithm,
    for<'db> A: db::RowLiteralSearch<db::Utf8Column<'db>>,
{
    let like = LikePattern::<A>::compile(pattern).unwrap();
    let mut scan = FullScan::new(text.row_count(), 1024);
    let mut scratch = QueryScratch::default();
    let mut matches = Vec::<RowId>::new();

    execute_like(&text, &mut scan, &like, &mut scratch, &mut matches);
    println!("{name:>10}: {matches:?}");
}

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

    let pattern = "%ana%";
    run::<StdSearch>("std", pattern, text);
    run::<db::Utf8Kmp>("kmp", pattern, text);
    run::<NaiveMixed>("naive", pattern, text);
    run::<BM>("bm", pattern, text);
    run::<TwoWay2>("two_way2", pattern, text);
    run::<FftStr1>("fft1", pattern, text);

    Ok(())
}
