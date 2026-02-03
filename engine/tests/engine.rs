use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use algos::StdSearch;
use engine::{execute, execute_all};
use like::compile_pattern;
use storage::{
    dataset::{load_text_table, DataSet},
    BumpArena,
};

fn make_temp_dir(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("bsc_engine_{}_{}", prefix, nanos));
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
}

#[test]
fn execute_matches_single_pattern() {
    let arena = BumpArena::new(4096);
    let dir = make_temp_dir("execute");
    let file_path = dir.join("sample.txt");
    write_file(&file_path, "hello");

    let table = load_text_table(&arena, &file_path).expect("load text table");
    let dataset = DataSet {
        tables: vec![table].into_boxed_slice(),
    };

    let pattern_str = "h%o";
    let pattern = compile_pattern::<StdSearch, _, _>(pattern_str, (), |_, pat| pat);

    let matches = execute(&pattern, &dataset);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].table, "sample.txt");
    assert_eq!(matches[0].row.data, "hello");
}

#[test]
fn execute_all_reports_pattern_index() {
    let arena = BumpArena::new(4096);
    let dir = make_temp_dir("execute_all");
    let file_path = dir.join("sample.txt");
    write_file(&file_path, "hello");

    let table = load_text_table(&arena, &file_path).expect("load text table");
    let dataset = DataSet {
        tables: vec![table].into_boxed_slice(),
    };

    let p0 = compile_pattern::<StdSearch, _, _>("h%o", (), |_, pat| pat);
    let p1 = compile_pattern::<StdSearch, _, _>("z%", (), |_, pat| pat);
    let patterns = vec![p0, p1];

    let matches = execute_all(&patterns, &dataset);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pattern_index, 0);
    assert_eq!(matches[0].table, "sample.txt");
}
