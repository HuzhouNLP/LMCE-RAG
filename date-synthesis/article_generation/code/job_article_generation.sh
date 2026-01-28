#!/bin/bash
#SBATCH --job-name=llama3_training
#SBATCH --nodes=1
#SBATCH --gres=gpu:4090d:1         # 申请1个GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2     # 每个任务使用2个CPU
#SBATCH --mem=200G            # 申请内存
#SBATCH --time=168:00:00      # 最大运行时间

# 获取任务 ID 和节点名a
JOB_ID=$SLURM_JOB_ID
NODE_NAME=$(hostname)

# 设置输出文件名
OUTPUT_FILE="article_generation-${JOB_ID}-${NODE_NAME}.log"

# 记录作业开始时间
echo "Job started at: $(date)" >> $OUTPUT_FILE

# 打印环境信息
hostname >> $OUTPUT_FILE
nvidia-smi >> $OUTPUT_FILE

# 激活虚拟环境
source ~/.bashrc
conda activate data

# 执行你的程序
srun bash run_all.sh >> $OUTPUT_FILE 2>&1
# 记录作业结束时间
echo "Job ended at: $(date)" >> $OUTPUT_FILE