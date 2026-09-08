#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
export RESULT_DIR="${RESULT_DIR:-$root/deepseek_results/linear_only_dynamic}"
exec "$root/benchmark/run_deepseek_dynamic.sh" linear
