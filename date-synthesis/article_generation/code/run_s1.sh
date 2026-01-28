#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR="${SCRIPT_DIR}/source"
CONFIG_OUTPUT_DIR="${SCRIPT_DIR}/../output/config"

# 是否启用 pair 模式（同一 LAW_BUCKET 内两两组合法条）
ENABLE_PAIR=true # 或 false

run_command() {
    local mode=$1
    local schema_file=$2
    local crime_names=$3
    local crime_name_idx=$4
    local json_idx=$5
    local pair_idx=${6:-}
    python "${SCRIPT_DIR}/s1_config.py" \
        --mode "$mode" \
        --data_for_complete "${SOURCE_DIR}/${schema_file}" \
        --crime_names "${SOURCE_DIR}/${crime_names}" \
        --crime_name_idx "$crime_name_idx" \
        --output_dir "${CONFIG_OUTPUT_DIR}" \
        --json_idx "$json_idx" \
        ${pair_idx:+--pair_idx "$pair_idx"} \
        --model_name deepseek-chat
}

# 定义schema文件和罪名配置
schema_file="crime.json"
crime_names="charge.json"
total_crimes=50              # 当前 charge.json 中共有 50 个罪名
configs_per_crime=5         # 每个罪名生成的 single-mode config 数量（single 和 pair 都会用到）

# single 部分用 0..(total_crimes*configs_per_crime-1)，pair 部分从 pair_offset 开始继续往后排
pair_offset=$((total_crimes * configs_per_crime))

# 循环遍历并发执行
for crime_name_idx in $(seq 0 $((total_crimes - 1))); do
    # single 模式：保持原有流程
    for iteration in $(seq 1 $configs_per_crime); do
        current_json_idx=$((crime_name_idx * configs_per_crime + iteration - 1))
        run_command "single" "$schema_file" "$crime_names" "$crime_name_idx" "$current_json_idx" &
    done
done

if [ "$ENABLE_PAIR" = true ]; then
    # pair 模式：按 LAW_BUCKET 遍历所有两两组合(i<j)
    bucket_starts=(0 10 20 30 40)
    bucket_ends=(9 19 29 39 49)

    pair_json_idx=$pair_offset

    for idx in "${!bucket_starts[@]}"; do
        start=${bucket_starts[$idx]}
        end=${bucket_ends[$idx]}
        for i in $(seq "$start" "$end"); do
            for j in $(seq $((i + 1)) "$end"); do
                # 每一对 (i, j) 也循环 configs_per_crime 次，生成多个不同的 pair config
                for iteration in $(seq 1 $configs_per_crime); do
                    run_command "pair" "$schema_file" "$crime_names" "$i" "$pair_json_idx" "$j" &
                    pair_json_idx=$((pair_json_idx + 1))
                done
            done
        done
    done
fi

wait
