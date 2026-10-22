#!/usr/bin/env bash
# scripts/run_item2.sh — run all four multi-seed jobs sequentially.
# Sequential (no n_jobs contention) is ~55 min total; faster than parallel
# was on this machine.
#
# Usage:
#   bash scripts/run_item2.sh               # 5 seeds (default)
#   bash scripts/run_item2.sh 42 123 456    # custom seeds
#
# Logs land at outputs/multiseed_<task>_<split>.log
# JSONs land at outputs/{,classify_}{random,scaffold}/multiseed_metrics.json

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${@:-42 123 456 789 1000}"
echo "[run_item2] seeds: $SEEDS"
echo "[run_item2] starting at $(date)"

run_one() {
  local task=$1 split=$2
  local log="outputs/multiseed_${task:0:3}_${split}.log"
  echo
  echo "===== $task / $split ====="
  date
  uv run python -u scripts/multi_seed_eval.py \
      --task "$task" --split "$split" --seeds $SEEDS 2>&1 | tee "$log"
}

run_one regression     random
run_one regression     scaffold
run_one classification random
run_one classification scaffold

echo
echo "[run_item2] all done at $(date)"
echo "[run_item2] summarising:"
uv run python scripts/summarize_item2.py
