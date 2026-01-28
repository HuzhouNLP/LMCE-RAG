#!/bin/bash

# Parameters
MODEL_NAME="deepseek-chat"
INPUT_DIR="/path/to/lkif/rageval/article_generation/output/config"
OUTPUT_DIR="/path/to/lkif/rageval/qar_generation/output/qra_multidoc"
NUMBER=780

run_python_script() {
    local model_name=$1
    local input_dir=$2
    local output_dir=$3
    local number=$4

    echo "Running multi-document QRA generation:"
    echo "  Model: $model_name"
    echo "  Input directory: $input_dir"
    echo "  Output directory: $output_dir"
    echo "  Number of pairs to generate: $number"
    
    python -u code/qra_pipeline_multi_doc.py \
        --model_name "$model_name" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --number "$number"
}

# Main execution
run_python_script "$MODEL_NAME" "$INPUT_DIR" "$OUTPUT_DIR" "$NUMBER"

echo "Multi-document QRA generation complete."