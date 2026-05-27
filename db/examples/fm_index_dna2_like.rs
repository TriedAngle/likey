use db::{
    DbBuilder, Dna2, Dna2TableBuilder, FmIndex, LikePattern, QueryScratch, RowId, execute_like,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reads = Dna2TableBuilder::new("reads");
    reads.push_str("ACGTACGT")?;
    reads.push_str("TTTACGTT")?;
    reads.push_str("GGGGGGGG")?;
    reads.push_str("AACGAAAA")?;

    let mut dbb = DbBuilder::new();
    let reads_id = dbb.add_dna2_table(reads)?;
    let db = dbb.freeze();

    let table = db.dna2_table(reads_id)?;
    let seq = table.sequence();

    // This LIKE verifier operates on packed DNA2 rows. The exact fragment ACG
    // becomes logical symbols [0, 1, 2], which the FM-index can use directly.
    let like = LikePattern::<Dna2>::compile("%ACG%")?;

    let fm = FmIndex::build(&seq)?;
    let mut candidates = fm
        .probe_longest_like_literal(&like, 1024)
        .expect("%ACG% has one exact DNA literal fragment");

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
