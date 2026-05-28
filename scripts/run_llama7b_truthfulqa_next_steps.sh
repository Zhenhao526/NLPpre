#!/usr/bin/env bash
set -euo pipefail

# Run follow-up TruthfulQA-MC checks on two GPUs:
# 1) vanilla only
# 2) DoLa high-layer bucket, aligned with the paper's TruthfulQA-MC command
# 3) DoLa low/mid/wide bucket sweeps for diagnosis
#
# Examples:
#   bash scripts/run_llama7b_truthfulqa_next_steps.sh
#   END_INDEX=100 MID_INDEX=50 bash scripts/run_llama7b_truthfulqa_next_steps.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-models/huggyllama-llama-7b}"
OUT_DIR="${OUT_DIR:-outputs/llama7b_next_steps}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-817}"
MID_INDEX="${MID_INDEX:-409}"
FORCE="${FORCE:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_DIR" "$LOG_DIR" configs/generated

python - <<'PY'
import os
from pathlib import Path

model_path = os.environ.get("MODEL_PATH", "models/huggyllama-llama-7b")
configs = {
    "llama7b_local_high": [16, 18, 20, 22, 24, 26, 28, 30],
    "llama7b_local_low": [0, 2, 4, 6, 8, 10, 12, 14],
    "llama7b_local_mid": [8, 10, 12, 14, 16, 18, 20, 22],
    "llama7b_local_wide": [0, 4, 8, 12, 16, 20, 24, 28],
}

template = """model_name: {model_path}
dataset_name: truthfulqa/truthful_qa
dataset_config: multiple_choice
split: validation
max_examples: 817
seed: 42
mature_layer: -1
candidate_premature_layers: {layers}
relative_top: 0.1
contrast_alpha: 1.0
temperature: 1.0
device: cuda
task_note: "LLaMA-7B TruthfulQA-MC diagnostic config: {name}."
"""

out_dir = Path("configs/generated")
for name, layers in configs.items():
    path = out_dir / f"hf_truthfulqa_{name}.yaml"
    path.write_text(template.format(model_path=model_path, layers=layers, name=name), encoding="utf-8")
    print(path)
PY

run_sharded() {
  local run_name="$1"
  local config_path="$2"
  local method="$3"
  local out0="$OUT_DIR/${run_name}_part0.csv"
  local out1="$OUT_DIR/${run_name}_part1.csv"
  local merged="$OUT_DIR/${run_name}.csv"
  local log0="$LOG_DIR/${run_name}_part0.log"
  local log1="$LOG_DIR/${run_name}_part1.log"

  if [[ "$FORCE" != "1" && -f "$merged" && -f "${merged%.csv}_summary.csv" ]]; then
    echo "[skip] $run_name already exists. Set FORCE=1 to rerun."
    return
  fi

  echo "[run] $run_name config=$config_path method=$method range=[$START_INDEX,$END_INDEX)"
  CUDA_VISIBLE_DEVICES="$GPU0" python scripts/run_hf_mc_eval.py \
    --config "$config_path" \
    --method "$method" \
    --start-index "$START_INDEX" \
    --end-index "$MID_INDEX" \
    --output "$out0" \
    > "$log0" 2>&1 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES="$GPU1" python scripts/run_hf_mc_eval.py \
    --config "$config_path" \
    --method "$method" \
    --start-index "$MID_INDEX" \
    --end-index "$END_INDEX" \
    --output "$out1" \
    > "$log1" 2>&1 &
  local pid1=$!

  set +e
  wait "$pid0"
  local status0=$?
  wait "$pid1"
  local status1=$?
  set -e

  if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
    echo "[error] $run_name failed: part0=$status0 part1=$status1"
    tail -n 80 "$log0" || true
    tail -n 80 "$log1" || true
    exit 1
  fi

  python scripts/merge_hf_mc_eval.py --inputs "$out0" "$out1" --output "$merged"
}

run_sharded "llama7b_vanilla_high" "configs/generated/hf_truthfulqa_llama7b_local_high.yaml" "vanilla"
run_sharded "llama7b_dola_high" "configs/generated/hf_truthfulqa_llama7b_local_high.yaml" "dola"
run_sharded "llama7b_dola_low" "configs/generated/hf_truthfulqa_llama7b_local_low.yaml" "dola"
run_sharded "llama7b_dola_mid" "configs/generated/hf_truthfulqa_llama7b_local_mid.yaml" "dola"
run_sharded "llama7b_dola_wide" "configs/generated/hf_truthfulqa_llama7b_local_wide.yaml" "dola"

python scripts/summarize_hf_mc_runs.py \
  --inputs \
    "$OUT_DIR/llama7b_vanilla_high_summary.csv" \
    "$OUT_DIR/llama7b_dola_high_summary.csv" \
    "$OUT_DIR/llama7b_dola_low_summary.csv" \
    "$OUT_DIR/llama7b_dola_mid_summary.csv" \
    "$OUT_DIR/llama7b_dola_wide_summary.csv" \
  --output "$OUT_DIR/summary_all.csv"

echo "[done] Summary:"
cat "$OUT_DIR/summary_all.csv"
