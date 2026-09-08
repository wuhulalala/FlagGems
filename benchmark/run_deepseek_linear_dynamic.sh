#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
result_dir="${RESULT_DIR:-$root/deepseek_results/linear_only_dynamic}"
cache_root="${CACHE_ROOT:-/workspace/flag_gems_benchmark_cache}"

mkdir -p "$result_dir" "$cache_root/huggingface" "$cache_root/triton" "$cache_root/torchinductor" "$cache_root/tmp"
export PYTHONPATH="$root/src:$root/benchmark"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLAGTREE_AABS=0
export HF_HOME="$cache_root/huggingface"
export HF_DATASETS_CACHE="$cache_root/huggingface/datasets"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TMPDIR="$cache_root/tmp"

for backend in eager flaggems trident; do
    /workspace/gemms_env/bin/python -u "$root/benchmark/benchmark_deepseek_dynamic.py" \
        --backend "$backend" --include linear --trident-scope dynamic \
        --task "${TASK:-humaneval}" --warmup-limit "${WARMUP_LIMIT:-5}" \
        --limit "${LIMIT:-5}" --warmup 1 --max-new-tokens "${TOKENS:-8}" \
        --cache-dir "$cache_root/huggingface/datasets" \
        --output "$result_dir/$backend.json" >"$result_dir/$backend.log" 2>&1
    echo "$backend completed"
done
