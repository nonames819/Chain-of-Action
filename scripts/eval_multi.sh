#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=5

# 添加你要处理的 checkpoint 文件夹路径
folders=(
    "exp_local/20250731/coa_push_button_rlbench_20250731003250/checkpoints"
)

# 最大并行任务数，根据机器性能设置
MAX_JOBS=4
job_count=0

for folder in "${folders[@]}"; do
    echo ">>> Searching in folder: $folder"

    # 自动提取任务名，例如从 coa_push_button_rlbench_20250731003250 提取 push_button
    parent_dir=$(basename "$(dirname "$folder")")
    task=$(echo "$parent_dir" | sed -E 's/^coa_([a-z_]+)_rlbench_.*/\1/')

    echo ">>> Extracted task: $task"

    for ckpt in "$folder"/*.pt; do
        if [ -f "$ckpt" ]; then
            echo ">>> Launching eval on $task checkpoint: $ckpt"

            # 并行执行评估脚本
            python scripts/eval.py task="$task" snapshot="$ckpt" &

            ((job_count++))
            if (( job_count % MAX_JOBS == 0 )); then
                echo ">>> Waiting for $MAX_JOBS jobs to finish..."
                wait
            fi
        else
            echo "No .pt files found in $folder"
        fi
    done
done

# 等待所有后台任务结束
wait
echo ">>> All evaluations completed."
