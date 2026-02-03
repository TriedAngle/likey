use algos::StringSearch;
use like::{like_match, Pattern};
use storage::dataset::{DataSet, Row};

#[derive(Debug, Clone, Copy)]
pub struct Match<'a> {
    pub table: &'a str,
    pub row: &'a Row<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchMatch<'a> {
    pub pattern_index: usize,
    pub table: &'a str,
    pub row: &'a Row<'a>,
}

pub fn execute<'p, 'd, S>(pattern: &Pattern<'p, S>, dataset: &'d DataSet<'d>) -> Vec<Match<'d>>
where
    S: StringSearch,
{
    let mut matches = Vec::new();

    for table in dataset.tables.iter() {
        let table_name = table.name.as_str();

        for row in table.rows.iter() {
            if like_match(pattern, row.data) {
                matches.push(Match {
                    table: table_name,
                    row,
                });
            }
        }
    }

    matches
}

pub fn execute_all<'p, 'd, S>(
    patterns: &[Pattern<'p, S>],
    dataset: &'d DataSet<'d>,
) -> Vec<BatchMatch<'d>>
where
    S: StringSearch,
{
    let mut matches = Vec::new();

    for (pattern_index, pattern) in patterns.iter().enumerate() {
        for table in dataset.tables.iter() {
            let table_name = table.name.as_str();

            for row in table.rows.iter() {
                if like_match(pattern, row.data) {
                    matches.push(BatchMatch {
                        pattern_index,
                        table: table_name,
                        row,
                    });
                }
            }
        }
    }

    matches
}
