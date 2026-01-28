#!/bin/bash
# 生成质量评估脚本 (generation)
# filepath: eval_generation.sh

START_TIME=$(date +%s)

REMOVE_REFUSAL=${1:-1}

PY_ARGS=(
  experiments/eval.py
  generation
  --llm_base_dir /path/to/lkif/Meta-Llama-3-8B-Instruct \
  --model_dir /path/to/lkif/model-dir \
  --encoder_dir "/path/to/lkif/model-dir/encoder/encoder.pt" \
  --dataset_dir /path/to/lkif/date-synthesis/qar_generation/results \
  --test_dataset QA-summary-long-test.json \
  --encoder_spec bge-large-en-v1.5 \
  --llm_type llama3 \
  --dataset_type law \
  --precomputed_embed_keys_path /path/to/lkif/date-synthesis/qar_generation/results/embedding/QA-summary-long-test_bge-large-en-v1.5_embd_key.npy \
  --precomputed_embed_values_path /path/to/lkif/date-synthesis/qar_generation/results/embedding/QA-summary-long-test_bge-large-en-v1.5_embd_value.npy \
  --save_dir /path/to/lkif/output-test-1b-long-50 \
  --eval_mode kb \
  --kb_size 50 \
  --max_new_tokens 512 \
  --seed 42
  

)

if [[ "${REMOVE_REFUSAL}" == "1" ]]; then
  PY_ARGS+=( --remove_sorry )
fi

python "${PY_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[eval_generation] Total wall time: ${ELAPSED}s"
