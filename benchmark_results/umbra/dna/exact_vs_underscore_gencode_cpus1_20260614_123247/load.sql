create table bench_data(key text, value text);
copy bench_data from '/work/benchmark_results/umbra/dna/exact_vs_underscore_gencode_cpus1_20260614_123247/umbra_input.csv' with (format csv, header true);
