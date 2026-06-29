#!/bin/bash

set -euo pipefail

# Move to project root: assumes this script is placed in the project root.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists.
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Keep logs.
LOG_DIR="logs/wselob_overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Log directory: $LOG_DIR"
echo

# Shared experiment settings.
NUM_LEVELS=50
HORIZON_SECONDS=30
SAMPLE_INTERVAL_SECONDS=1
CLIP_TICKS=10
TICK_SIZE_RAW=5
MAX_DAYS=-1

# Model training settings.
BATCH_SIZE=8192
EVAL_BATCH_SIZE=32768
MAX_EPOCHS=50
PATIENCE=6

# Stocks to process.
STOCKS=(
  "PZU:data/raw/WSELOB-2017/orders/PZU_lob_2017_zlib.h5"
)

for STOCK_ENTRY in "${STOCKS[@]}"; do
  STOCK="${STOCK_ENTRY%%:*}"
  INPUT_FILE="${STOCK_ENTRY#*:}"

  DATASET_NAME="WSELOB_${STOCK}_h30_L50_full_tick5"
  PROCESSED_DIR="data/processed/${DATASET_NAME}"
  RESULTS_DIR="results/${DATASET_NAME}"

  echo "============================================================"
  echo "Starting experiment for ${STOCK}"
  echo "Input file:     ${INPUT_FILE}"
  echo "Processed dir:  ${PROCESSED_DIR}"
  echo "Results dir:    ${RESULTS_DIR}"
  echo "============================================================"
  echo

  echo "Step 1/5: Converting WSELOB data for ${STOCK}"
  python -u src/convert_wselob.py \
    --input "${INPUT_FILE}" \
    --output "${PROCESSED_DIR}" \
    --num-levels "${NUM_LEVELS}" \
    --sample-interval-seconds "${SAMPLE_INTERVAL_SECONDS}" \
    --horizon-seconds "${HORIZON_SECONDS}" \
    --clip-ticks "${CLIP_TICKS}" \
    --tick-size-raw "${TICK_SIZE_RAW}" \
    --max-days "${MAX_DAYS}" \
    2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_convert.log"

  echo
  echo "Step 2/5: Running empirical baseline for ${STOCK}"
  python -u src/train_empirical.py \
    --processed-dir "${PROCESSED_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_empirical.log"

  echo
  echo "Step 3/5: Running logistic regression baseline for ${STOCK}"
  python -u src/train_logistic.py \
    --processed-dir "${PROCESSED_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --max-epochs 30 \
    --patience 5 \
    2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_logistic.log"

  echo
  echo "Step 4/5: Running standard MLP baseline for ${STOCK}"
  python -u src/train_mlp.py \
    --processed-dir "${PROCESSED_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_mlp.log"

  echo
  echo "Step 5/5: Running reduced spatial neural network for ${STOCK}"
  python -u src/train_spatial.py \
    --processed-dir "${PROCESSED_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_spatial.log"

  echo
  echo "Finished experiment for ${STOCK}"
  echo
done

echo "============================================================"
echo "All WSELOB overnight experiments finished."
echo "Logs saved in: ${LOG_DIR}"
echo "============================================================"