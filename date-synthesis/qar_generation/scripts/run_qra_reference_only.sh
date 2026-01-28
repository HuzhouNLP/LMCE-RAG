#!/bin/bash
#run_qra_reference_only.sh
# --------- 参数区，可按需修改 ----------
MODEL_NAME="deepseek-chat"
INPUT_DIR="/path/to/lkif/qar_generation/output/QA-test"
DOC_DIR="/path/to/lkif/article_generation/output/doc-test"
OUTPUT_DIR="/path/to/lkif/qar_generation/output/QA_ref_only-test"
PROMPT_PATH="code/prompts.jsonl"
PYTHON_BIN="python"
# ----------------------------------------

set -euo pipefail

echo "[INFO] 开始执行 question-reference 对齐，输入目录: ${INPUT_DIR}"
${PYTHON_BIN} -u code/qra_reference_only.py \
    --model_name "${MODEL_NAME}" \
    --input_dir "${INPUT_DIR}" \
    --doc_dir "${DOC_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_path "${PROMPT_PATH}"

echo "[INFO] question-reference 对齐完成，结果已写入 ${OUTPUT_DIR}"
