#!/usr/bin/env bash
set -euo pipefail

# Run the official DoLa TruthfulQA-MC baseline and high-layer DoLa commands.
# The official README uses:
#   baseline: no --early-exit-layers
#   DoLa: --early-exit-layers 16,18,20,22,24,26,28,30,32

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOLA_DIR="${DOLA_DIR:-$HOME/third_party_DoLa}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/huggyllama-llama-7b}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/official_truthfulqa}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/official_dola_truthfulqa}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/logs}"
GPU="${GPU:-0}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-22}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

mkdir -p "$DATA_DIR" "$OUT_DIR" "$LOG_DIR"

if [[ ! -d "$DOLA_DIR/.git" ]]; then
  git clone https://github.com/voidism/DoLa.git "$DOLA_DIR"
fi

cd "$DOLA_DIR"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  python -m pip install -r requirements.txt
  python -m pip install pandas tqdm sentencepiece protobuf accelerate
  if [[ -d transformers-4.28.1 ]]; then
    python -m pip install -e transformers-4.28.1
  fi
fi

if [[ ! -f "$DATA_DIR/TruthfulQA.csv" ]]; then
  curl -L https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv \
    -o "$DATA_DIR/TruthfulQA.csv"
fi

CUDA_VISIBLE_DEVICES="$GPU" python tfqa_mc_eval.py \
  --model-name "$MODEL_PATH" \
  --data-path "$DATA_DIR" \
  --output-path "$OUT_DIR/tfqa_mc_baseline.json" \
  --num-gpus 1 \
  --max_gpu_memory "$MAX_GPU_MEMORY" \
  > "$LOG_DIR/official_tfqa_mc_baseline.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" python tfqa_mc_eval.py \
  --model-name "$MODEL_PATH" \
  --early-exit-layers 16,18,20,22,24,26,28,30,32 \
  --data-path "$DATA_DIR" \
  --output-path "$OUT_DIR/tfqa_mc_dola_high.json" \
  --num-gpus 1 \
  --max_gpu_memory "$MAX_GPU_MEMORY" \
  > "$LOG_DIR/official_tfqa_mc_dola_high.log" 2>&1

cd "$REPO_ROOT"
python scripts/summarize_official_dola_json.py \
  --inputs "$OUT_DIR/tfqa_mc_baseline.json" "$OUT_DIR/tfqa_mc_dola_high.json" \
  --output "$OUT_DIR/tfqa_mc_summary.csv"

cat "$OUT_DIR/tfqa_mc_summary.csv"
