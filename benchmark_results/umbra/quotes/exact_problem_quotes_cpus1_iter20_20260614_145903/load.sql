create table bench_data(key text, value text);
copy bench_data from '/work/benchmark_results/umbra/quotes/exact_problem_quotes_cpus1_iter20_20260614_145903/umbra_input.csv' with (format csv, header true);
