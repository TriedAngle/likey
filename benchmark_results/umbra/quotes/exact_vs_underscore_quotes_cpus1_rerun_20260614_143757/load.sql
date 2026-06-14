create table bench_data(key text, value text);
copy bench_data from '/work/benchmark_results/umbra/quotes/exact_vs_underscore_quotes_cpus1_rerun_20260614_143757/umbra_input.csv' with (format csv, header true);
