use db::{
    DbBuilder, Dna2NaiveWildcard, Dna2TableBuilder, FmIndex, LikePattern, QueryScratch, RowId,
    execute_like,
};

const FM_WILDCARD: u8 = 0xFF;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reads = Dna2TableBuilder::new("reads");
    reads.push_str("ACGT")?; // A C G T
    reads.push_str("AGGT")?; // A G G T
    reads.push_str("ATGT")?; // A T G T
    reads.push_str("AAAA")?;

    let mut dbb = DbBuilder::new();
    let reads_id = dbb.add_dna2_table(reads)?;
    let db = dbb.freeze();

    let table = db.dna2_table(reads_id)?;
    let seq = table.sequence();

    // The verifier can interpret `_` inside a DNA literal directly.
    let like = LikePattern::<Dna2NaiveWildcard>::compile("%A_G%")?;

    // Because that literal contains a wildcard, `longest_indexable_literal()` is
    // intentionally None. But the FM-index has an optional wildcard probe for
    // candidate generation. External DNA symbols are A=0, C=1, G=2, T=3.
    let fm = FmIndex::build(&seq)?;
    let mut candidates = fm.probe_with_wildcard(&[0, FM_WILDCARD, 2], FM_WILDCARD, 1024);

    let mut scratch = QueryScratch::default();
    let mut matches = Vec::<RowId>::new();
    let stats = execute_like(&seq, &mut candidates, &like, &mut scratch, &mut matches);

    println!("matching row ids: {matches:?}");
    for row in &matches {
        println!("row {row}: {}", seq.row_to_ascii_string(*row));
    }
    println!("stats: {stats:?}");

    Ok(())
}
