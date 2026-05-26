use db::{
    Column, DbBuilder, FmIndex, FsstTableBuilder, FullScan, LikePattern, NaiveMixed, QueryScratch,
    RowId, TrigramIndex, execute_like,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut docs = FsstTableBuilder::new("docs_fsst");
    docs.push_str("banana");
    docs.push_str("bandana");
    docs.push_str("apple");
    docs.push_str("canary");

    let mut dbb = DbBuilder::new();
    let docs_id = dbb.add_fsst_table(docs)?;
    let db = dbb.freeze();

    let table = db.fsst_table(docs_id)?;
    let text = table.text();

    println!(
        "rows={}, compressed={} bytes, uncompressed={} bytes, ratio={:?}",
        text.row_count(),
        text.compressed_bytes(),
        text.uncompressed_bytes(),
        text.compression_ratio()
    );

    // FSST rows are decoded for the row verifier in this baseline path.
    let like = LikePattern::<NaiveMixed>::compile("%ana%")?;
    let mut scan = FullScan::new(text.row_count(), 1024);
    let mut scratch = QueryScratch::default();
    let mut matches = Vec::<RowId>::new();
    execute_like(&text, &mut scan, &like, &mut scratch, &mut matches);
    println!("full-scan matches: {matches:?}");

    // Generic indexes still work because FsstColumn exposes decoded u8 symbols.
    let trigram = TrigramIndex::build(&text);
    let mut tri_probe = trigram.probe(*b"ana");
    let mut tri_matches = Vec::<RowId>::new();
    execute_like(&text, &mut tri_probe, &like, &mut scratch, &mut tri_matches);
    println!("trigram candidates + verify matches: {tri_matches:?}");

    let fm = FmIndex::build(&text)?;
    let mut fm_probe = fm.probe(b"ana", 1024);
    let mut fm_matches = Vec::<RowId>::new();
    execute_like(&text, &mut fm_probe, &like, &mut scratch, &mut fm_matches);
    println!("fm candidates + verify matches: {fm_matches:?}");

    Ok(())
}
