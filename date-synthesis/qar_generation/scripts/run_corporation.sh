#!/bin/bash
#run_corporation.sh

OUTPUT_DIR="/path/to/lkif/date-synthesis/qar_generation/results"  # Directory to store the output files
SINGLE_DOC_DIR="/path/to/lkif/date-synthesis/qar_generation/output/QA-16814"  # Single-doc QRA JSON configs
DOC_ROOT_DIR="/path/to/lkif/date-synthesis/article_generation/output/doc"  # Original documents
TYPE='qra'            # Type of corporation, either 'qra' or 'doc'

# Function: Run the Python script
run_python_script() {
    local output_dir=$1
    local single_doc_dir=$2
    local doc_root_dir=$3
    local type=$4

    echo "Running $type corporation for single_doc_dir: $single_doc_dir, doc_root_dir: $doc_root_dir, output directory: $output_dir"
    python -u code/data_processing/${type}_corporation_simple.py \
        --single_doc_dir "$single_doc_dir" \
        --doc_root_dir "$doc_root_dir" \
        --output_dir "$output_dir"
}

run_python_script "$OUTPUT_DIR" "$SINGLE_DOC_DIR" "$DOC_ROOT_DIR" "$TYPE"

echo "$TYPE corporation complete."