#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_all_benchmarks.py \
  --only-mode matcher-multi-percent \
  --iterations 10 \
  --warmups 1 \
  "$@"
