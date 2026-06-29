#!/bin/bash

set -euo pipefail

# Assumes this script is placed in the project root.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if present.
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Prevent macOS from sleeping while running.
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu &
  CAFFEINATE_PID=$!
  trap 'kill "$CAFFEINATE_PID" 2>/dev/null || true' EXIT
fi

LOG_DIR="logs/convert_and_full_spatial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Log directory: $LOG_DIR"
echo

# ---------------------------------------------------------------------
# Dataset conversion settings
# ---------------------------------------------------------------------

NUM_LEVELS=50
SAMPLE_INTERVAL_SECONDS=1
HORIZON_SECONDS=30
CLIP_TICKS=10
TICK_SIZE_RAW=5
MAX_DAYS=-1

SAVE_DENSE_GRID=true
DENSE_MAX_OFFSET=50
DENSE_CHUNK_SIZE=250000

# ---------------------------------------------------------------------
# Full spatial training settings
# ---------------------------------------------------------------------

BATCH_SIZE=4096
EVAL_BATCH_SIZE=32768
FULL_EVAL_BATCH_SIZE=2048
CHUNK_SIZE=1048576

GLOBAL_DEPTH=50
LOCAL_WINDOW=2

LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
MAX_EPOCHS=50
PATIENCE=6
SEED=42

# ---------------------------------------------------------------------
# Remaining stocks
# KGHM is left out because it has already been converted and trained.
# ---------------------------------------------------------------------

STOCKS=(
  "PEKAO:data/raw/WSELOB-2017/orders/PEKAO_lob_2017_zlib.h5"
  "PKNORLEN:data/raw/WSELOB-2017/orders/PKNORLEN_lob_2017_zlib.h5"
  "PKOBP:data/raw/WSELOB-2017/orders/PKOBP_lob_2017_zlib.h5"
  "PZU:data/raw/WSELOB-2017/orders/PZU_lob_2017_zlib.h5"
)

for STOCK_ENTRY in "${STOCKS[@]}"; do
  STOCK="${STOCK_ENTRY%%:*}"
  INPUT_FILE="${STOCK_ENTRY#*:}"

  DATASET_NAME="WSELOB_${STOCK}_h30_L50_full_tick5"
  PROCESSED_DIR="data/processed/${DATASET_NAME}"
  RESULTS_DIR="results/${DATASET_NAME}_full_spatial"

  CONVERT_LOG="${LOG_DIR}/${DATASET_NAME}_convert.log"
  TRAIN_LOG="${LOG_DIR}/${DATASET_NAME}_full_spatial.log"

  echo "============================================================"
  echo "Starting conversion and full spatial training for ${STOCK}"
  echo "Input file:     ${INPUT_FILE}"
  echo "Processed dir:  ${PROCESSED_DIR}"
  echo "Results dir:    ${RESULTS_DIR}"
  echo "============================================================"
  echo

  if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file does not exist:"
    echo "${INPUT_FILE}"
    exit 1
  fi

  echo "Step 1/2: Converting WSELOB data for ${STOCK}"
  echo "Log file: ${CONVERT_LOG}"
  echo

  python -u src/convert_wselob.py \
    --input "${INPUT_FILE}" \
    --output "${PROCESSED_DIR}" \
    --num-levels "${NUM_LEVELS}" \
    --sample-interval-seconds "${SAMPLE_INTERVAL_SECONDS}" \
    --horizon-seconds "${HORIZON_SECONDS}" \
    --clip-ticks "${CLIP_TICKS}" \
    --tick-size-raw "${TICK_SIZE_RAW}" \
    --max-days "${MAX_DAYS}" \
    --save-dense-grid \
    --dense-max-offset "${DENSE_MAX_OFFSET}" \
    --dense-chunk-size "${DENSE_CHUNK_SIZE}" \
    2>&1 | tee "${CONVERT_LOG}"

  echo
  echo "Step 2/2: Training full spatial NN for ${STOCK}"
  echo "Log file: ${TRAIN_LOG}"
  echo

  python -u src/train_full_spatial.py \
    --processed-dir "${PROCESSED_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --full-eval-batch-size "${FULL_EVAL_BATCH_SIZE}" \
    --chunk-size "${CHUNK_SIZE}" \
    --global-depth "${GLOBAL_DEPTH}" \
    --local-window "${LOCAL_WINDOW}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --max-epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    --seed "${SEED}" \
    2>&1 | tee "${TRAIN_LOG}"

  echo
  echo "Finished ${STOCK}"
  echo
done

echo "============================================================"
echo "All remaining-stock full spatial experiments finished."
echo "Logs saved in: ${LOG_DIR}"
echo "============================================================"