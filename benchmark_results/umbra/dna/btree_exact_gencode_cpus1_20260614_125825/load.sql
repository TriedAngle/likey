create table bench_data(key text, value text);
copy bench_data from '/work/benchmark_results/umbra/dna/btree_exact_gencode_cpus1_20260614_125825/umbra_input.csv' with (format csv, header true);
create index bench_data_value_idx on bench_data(value);
