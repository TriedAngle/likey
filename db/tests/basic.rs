use db::{
    AcceptAll, Column, DbBuilder, Dna2TableBuilder, DnaBase, FsstTableBuilder, FullScan,
    LenConstraint, QueryScratch, RowId, SortedRowsProbe, Utf8TableBuilder, execute_like,
};

#[test]
fn utf8_storage_roundtrip() {
    let mut dbb = DbBuilder::new();
    let mut t = Utf8TableBuilder::new("docs");
    t.push_str("hello");
    t.push_str("world");
    let id = dbb.add_utf8_table(t).unwrap();
    let db = dbb.freeze();

    let table = db.utf8_table(id).unwrap();
    let col = table.text();
    assert_eq!(col.row_count(), 2);
    assert_eq!(col.row_str(0).unwrap(), "hello");
    assert_eq!(col.row_str(1).unwrap(), "world");
    assert_eq!(col.logical_len(0), 5);
}

#[test]
fn dna2_storage_roundtrip() {
    let mut dbb = DbBuilder::new();
    let mut t = Dna2TableBuilder::new("reads");
    t.push_str("ACGT").unwrap();
    t.push_str("TTAA").unwrap();
    let id = dbb.add_dna2_table(t).unwrap();
    let db = dbb.freeze();

    let table = db.dna2_table(id).unwrap();
    let col = table.sequence();
    assert_eq!(col.row_count(), 2);
    assert_eq!(col.row_to_ascii_string(0), "ACGT");
    assert_eq!(col.row_to_ascii_string(1), "TTAA");
    assert_eq!(col.base_code_at(0, 0), DnaBase::A.code());
    assert_eq!(col.base_code_at(0, 3), DnaBase::T.code());
}

#[test]
fn fsst_storage_roundtrip() {
    let mut dbb = DbBuilder::new();
    let mut t = FsstTableBuilder::new("docs_fsst");
    t.push_str("banana");
    t.push_str("bandana");
    let id = dbb.add_fsst_table(t).unwrap();
    let db = dbb.freeze();

    let table = db.fsst_table(id).unwrap();
    let col = table.text();
    assert_eq!(col.row_count(), 2);
    assert_eq!(col.row(0).bytes(), b"banana");
    assert_eq!(col.row(1).bytes(), b"bandana");
    assert_eq!(col.logical_len(0), 6);
}

#[test]
fn execute_full_scan_with_length_filter() {
    let mut dbb = DbBuilder::new();
    let mut t = Utf8TableBuilder::new("docs");
    t.push_str("a");
    t.push_str("abc");
    t.push_str("abcde");
    let id = dbb.add_utf8_table(t).unwrap();
    let db = dbb.freeze();

    let col = db.utf8_table(id).unwrap().text();
    let verifier = AcceptAll::new(LenConstraint::between(2, 4));
    let mut scan = FullScan::new(col.row_count(), 2);
    let mut scratch = QueryScratch::default();
    let mut rows = Vec::<RowId>::new();
    let stats = execute_like(&col, &mut scan, &verifier, &mut scratch, &mut rows);

    assert_eq!(rows, vec![1]);
    assert_eq!(stats.candidate_rows_seen, 3);
    assert_eq!(stats.rows_after_len_filter, 1);
    assert_eq!(stats.rows_matched, 1);
}

#[test]
fn execute_sorted_rows_probe() {
    let mut dbb = DbBuilder::new();
    let mut t = Utf8TableBuilder::new("docs");
    t.push_str("one");
    t.push_str("two");
    t.push_str("three");
    let id = dbb.add_utf8_table(t).unwrap();
    let db = dbb.freeze();

    let col = db.utf8_table(id).unwrap().text();
    let candidates = [0, 2];
    let mut probe = SortedRowsProbe::new(&candidates, 1);
    let verifier = AcceptAll::default();
    let mut scratch = QueryScratch::default();
    let mut rows = Vec::<RowId>::new();
    execute_like(&col, &mut probe, &verifier, &mut scratch, &mut rows);

    assert_eq!(rows, vec![0, 2]);
}
