#!/bin/bash
#run_qra_single_doc.sh

# Parameters
MODEL_NAME="deepseek-chat"  # Name of the OpenAI model to use
INPUT_DIR="/path/to/lkif/article_generation/output/config"  # Directory containing input JSON files
OUTPUT_DIR="/path/to/lkif/qar_generation/output/QA-copy"  # Directory to store the generated QRA files

# Function: Run the Python script
run_python_script() {
    local model_name=$1
    local input_dir=$2
    local output_dir=$3

    echo "Processing all documents in $input_dir"
    python -u code/qra_pipeline_single_doc.py \
        --model_name "$model_name" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir"
}

# Main execution
run_python_script "$MODEL_NAME" "$INPUT_DIR" "$OUTPUT_DIR"

echo "single doc QRA generation complete."