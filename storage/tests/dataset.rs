use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use storage::{
    BumpArena,
    dataset::{
        Source, SourceKind, infer_source_kind, load_dataset, load_dataset_from_paths,
        load_fasta_table, load_text_table,
    },
};

fn make_temp_dir(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("bsc_{}_{}", prefix, nanos));
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
}

#[test]
fn load_text_table_basic() {
    let arena = BumpArena::new(4096);
    let dir = make_temp_dir("text_table");
    let file_path = dir.join("sample.txt");
    write_file(&file_path, "hello world");

    let table = load_text_table(&arena, &file_path).expect("load text table");

    assert_eq!(table.name, "sample.txt");
    assert_eq!(table.rows.len(), 1);
    assert_eq!(table.rows[0].id, "sample.txt");
    assert_eq!(table.rows[0].desc, "");
    assert_eq!(table.rows[0].data, "hello world");
}

#[test]
fn load_fasta_table_basic() {
    let arena = BumpArena::new(4096);
    let dir = make_temp_dir("fasta_table");
    let file_path = dir.join("sample.fasta");
    let fasta = ">seq1 Human Gene\nATGC\nATGC\n>seq2\nGGCC\n";
    write_file(&file_path, fasta);

    let table = load_fasta_table(&arena, &file_path).expect("load fasta table");

    assert_eq!(table.name, "sample.fasta");
    assert_eq!(table.rows.len(), 2);
    assert_eq!(table.rows[0].id, "seq1");
    assert_eq!(table.rows[0].desc, "Human Gene");
    assert_eq!(table.rows[0].data, "ATGCATGC");
    assert_eq!(table.rows[1].id, "seq2");
    assert_eq!(table.rows[1].desc, "");
    assert_eq!(table.rows[1].data, "GGCC");
}

#[test]
fn load_dataset_mixed_sources() {
    let arena = BumpArena::new(8192);
    let dir = make_temp_dir("mixed_dataset");

    let text_path = dir.join("sample.txt");
    write_file(&text_path, "hello world");

    let fasta_path = dir.join("sample.fa");
    let fasta = ">seq1\nACGT\n";
    write_file(&fasta_path, fasta);

    let sources = vec![
        Source {
            path: text_path.clone(),
            kind: SourceKind::Text,
        },
        Source {
            path: fasta_path.clone(),
            kind: SourceKind::Fasta,
        },
    ];

    let dataset = load_dataset(&arena, &sources).expect("load dataset");

    assert_eq!(dataset.tables.len(), 2);
    assert_eq!(dataset.tables[0].name, "sample.txt");
    assert_eq!(dataset.tables[1].name, "sample.fa");
    assert_eq!(dataset.tables[0].rows.len(), 1);
    assert_eq!(dataset.tables[1].rows.len(), 1);
}

#[test]
fn infer_source_kind_by_extension() {
    let fasta = Path::new("/tmp/sample.fasta");
    let fa = Path::new("/tmp/sample.fa");
    let fna = Path::new("/tmp/sample.fna");
    let txt = Path::new("/tmp/sample.txt");

    assert_eq!(infer_source_kind(fasta), SourceKind::Fasta);
    assert_eq!(infer_source_kind(fa), SourceKind::Fasta);
    assert_eq!(infer_source_kind(fna), SourceKind::Fasta);
    assert_eq!(infer_source_kind(txt), SourceKind::Text);
}

#[test]
fn load_dataset_from_paths_infers_kind() {
    let arena = BumpArena::new(8192);
    let dir = make_temp_dir("infer_dataset");

    let text_path = dir.join("sample.txt");
    write_file(&text_path, "hello world");

    let fasta_path = dir.join("sample.fasta");
    let fasta = ">seq1\nACGT\n";
    write_file(&fasta_path, fasta);

    let paths = vec![text_path.clone(), fasta_path.clone()];
    let dataset = load_dataset_from_paths(&arena, &paths).expect("load dataset from paths");

    assert_eq!(dataset.tables.len(), 2);
    assert_eq!(dataset.tables[0].name, "sample.txt");
    assert_eq!(dataset.tables[1].name, "sample.fasta");
    assert_eq!(dataset.tables[0].rows.len(), 1);
    assert_eq!(dataset.tables[1].rows.len(), 1);
}
