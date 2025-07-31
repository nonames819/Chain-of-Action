#!/bin/bash

# 使用方法: ./script.sh [gpu_start_number]
# 例如: ./script.sh 0  (从GPU 0开始使用)
#      ./script.sh 2  (从GPU 2开始使用)

GPU_START=${1:-0}  # 默认从GPU 0开始
SESSION="eval_session"

# 检查会话是否已存在，如果存在则先删除
tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

tmux new-session -d -s $SESSION  # 创建后台新会话

folders=(
    "exp_local/20250731/coa_push_button_rlbench_20250731003250/checkpoints"
)

window=0
pane_count=0
current_gpu=$GPU_START

for folder in "${folders[@]}"; do
    # 检查文件夹是否存在
    if [ ! -d "$folder" ]; then
        echo "警告: 文件夹 $folder 不存在，跳过"
        continue
    fi
    
    # 取任务名
    parent_dir=$(basename "$(dirname "$folder")")
    task=$(echo "$parent_dir" | sed -E 's/^coa_([a-z_]+)_rlbench_.*/\1/')
    
    echo "处理任务: $task"

    for ckpt in "$folder"/*.pt; do
        if [ -f "$ckpt" ]; then
            ckpt_name=$(basename "$ckpt")
            echo "  - 检查点: $ckpt_name (GPU: $current_gpu)"
            
            # 构建命令，包含GPU设置
            cmd="CUDA_VISIBLE_DEVICES=$current_gpu python scripts/eval.py task=$task snapshot=$ckpt"
            
            # 第一个任务放到第一个pane
            if [ $pane_count -eq 0 ]; then
                # 在第一个面板运行
                tmux send-keys -t ${SESSION}:$window.0 "$cmd" C-m
            else
                # 新建面板（竖直分割）
                tmux split-window -t ${SESSION}:$window -v
                # 运行命令
                tmux send-keys -t ${SESSION}:$window.$pane_count "$cmd" C-m
            fi
            
            pane_count=$((pane_count + 1))
            current_gpu=$((current_gpu + 1))
            
            # 可选：限制最大GPU数量，循环使用
            # if [ $current_gpu -gt 7 ]; then
            #     current_gpu=$GPU_START
            # fi
        fi
    done
done

if [ $pane_count -eq 0 ]; then
    echo "错误: 没有找到任何 .pt 文件"
    tmux kill-session -t $SESSION
    exit 1
fi

echo "创建了 $pane_count 个面板，使用GPU $GPU_START 到 $((current_gpu-1))"

# 平铺所有面板以便更好地查看
tmux select-layout -t ${SESSION}:$window tiled

# 选回第一个面板
tmux select-pane -t ${SESSION}:$window.0

echo "连接到tmux会话: $SESSION"
echo "使用 'tmux detach' 或 Ctrl+b d 来分离会话"
echo "使用 'tmux attach-session -t $SESSION' 来重新连接"

# 连接到tmux会话
tmux attach-session -t $SESSION