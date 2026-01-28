#!/bin/bash

MODEL_NAME=${1:-"bge-large-en-v1.5"}
DATASET_PATH=${2:-"date-synthesis/qar_generation/results/QA-summary-long-test.json"}
OUTPUT_PATH=${3:-"date-synthesis/qar_generation/results/embedding"}
DATASET_NAME=${4:-"QA-summary-long-test"}

# 切到项目根目录（兼容从任意位置调用，包括集群作业）
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

PYTHON=python

echo "[kb_embeddings] Configuration:"
echo "  MODEL_NAME   = ${MODEL_NAME}"
echo "  DATASET_PATH = ${DATASET_PATH}"
echo "  OUTPUT_PATH  = ${OUTPUT_PATH}"
echo "  DATASET_NAME = ${DATASET_NAME}"
echo
echo "[kb_embeddings] Start generating KB embeddings ..."

$PYTHON dataset_generation/generate_kb_embeddings.py \
  --model_name "${MODEL_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${OUTPUT_PATH}"

echo "[kb_embeddings] Done."