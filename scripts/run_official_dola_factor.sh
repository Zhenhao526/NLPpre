#!/usr/bin/env bash
set -euo pipefail

# Run official DoLa FACTOR experiments.
# Default mode is a quick debug run. Use RUN_FULL=1 for full Wiki/News FACTOR.
#
# Examples:
#   bash scripts/run_official_dola_factor.sh
#   RUN_FULL=1 bash scripts/run_official_dola_factor.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOLA_DIR="${DOLA_DIR:-$HOME/third_party_DoLa}"
FACTOR_DIR="${FACTOR_DIR:-$HOME/factor_data}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/huggyllama-llama-7b}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/official_dola_factor}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/logs}"
GPU_BASELINE="${GPU_BASELINE:-0}"
GPU_DOLA="${GPU_DOLA:-1}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-22}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
RUN_FULL="${RUN_FULL:-0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -d "$DOLA_DIR/.git" ]]; then
  git clone https://github.com/voidism/DoLa.git "$DOLA_DIR"
fi

if [[ ! -d "$FACTOR_DIR/.git" ]]; then
  git clone https://github.com/AI21Labs/factor.git "$FACTOR_DIR"
fi

cd "$DOLA_DIR"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  python -m pip install -r requirements.txt
  python -m pip install pandas tqdm sentencepiece protobuf accelerate
  if [[ -d transformers-4.28.1 ]]; then
    python -m pip install -e transformers-4.28.1
  fi
fi

find_factor_file() {
  local name="$1"
  local path
  path="$(find "$FACTOR_DIR" -type f -name "$name" | head -n 1 || true)"
  if [[ -z "$path" ]]; then
    echo "[error] Cannot find $name under $FACTOR_DIR" >&2
    exit 1
  fi
  echo "$path"
}

WIKI_FACTOR="$(find_factor_file wiki_factor.csv)"
NEWS_FACTOR="$(find_factor_file news_factor.csv)"

run_factor_pair() {
  local dataset_name="$1"
  local data_path="$2"
  local extra_args=()
  if [[ "$RUN_FULL" != "1" ]]; then
    extra_args=(--debug)
  fi

  local baseline_out="$OUT_DIR/${dataset_name}_baseline.json"
  local dola_out="$OUT_DIR/${dataset_name}_dola.json"
  local baseline_log="$LOG_DIR/factor_${dataset_name}_baseline.log"
  local dola_log="$LOG_DIR/factor_${dataset_name}_dola.log"

  echo "[run] FACTOR $dataset_name baseline"
  CUDA_VISIBLE_DEVICES="$GPU_BASELINE" python factor_eval.py \
    --model-name "$MODEL_PATH" \
    --early-exit-layers -1 \
    --data-path "$data_path" \
    --output-path "$baseline_out" \
    --num-gpus 1 \
    --max_gpu_memory "$MAX_GPU_MEMORY" \
    "${extra_args[@]}" \
    > "$baseline_log" 2>&1 &
  local pid_baseline=$!

  echo "[run] FACTOR $dataset_name DoLa"
  CUDA_VISIBLE_DEVICES="$GPU_DOLA" python factor_eval.py \
    --model-name "$MODEL_PATH" \
    --early-exit-layers 0,2,4,6,8,10,12,14,32 \
    --data-path "$data_path" \
    --output-path "$dola_out" \
    --num-gpus 1 \
    --max_gpu_memory "$MAX_GPU_MEMORY" \
    "${extra_args[@]}" \
    > "$dola_log" 2>&1 &
  local pid_dola=$!

  set +e
  wait "$pid_baseline"
  local status_baseline=$?
  wait "$pid_dola"
  local status_dola=$?
  set -e

  if [[ "$status_baseline" -ne 0 || "$status_dola" -ne 0 ]]; then
    echo "[error] FACTOR $dataset_name failed: baseline=$status_baseline dola=$status_dola"
    tail -n 100 "$baseline_log" || true
    tail -n 100 "$dola_log" || true
    exit 1
  fi
}

run_factor_pair "wiki" "$WIKI_FACTOR"
run_factor_pair "news" "$NEWS_FACTOR"

cd "$REPO_ROOT"

mapfile -t FACTOR_JSONS < <(find "$OUT_DIR" -type f -name "*.json" | sort)
if [[ "${#FACTOR_JSONS[@]}" -eq 0 ]]; then
  echo "[error] No FACTOR JSON outputs found in $OUT_DIR" >&2
  exit 1
fi
python scripts/summarize_official_dola_json.py \
  --inputs "${FACTOR_JSONS[@]}" \
  --output "$OUT_DIR/factor_summary.csv"

cat "$OUT_DIR/factor_summary.csv"
