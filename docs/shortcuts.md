"reach_target",
"press_switch",
"pick_up_cup",
"open_drawer",
"stack_wine",
"open_box",
"sweep_to_dustpan",
"turn_tap",
"push_button",
"take_lid_off_saucepan"

* train
CUDA_VISIBLE_DEVICES=4 python scripts/train.py task=task_name 

CUDA_VISIBLE_DEVICES=4 python scripts/train.py task=press_switch
CUDA_VISIBLE_DEVICES=4 python scripts/train.py task=push_button 
CUDA_VISIBLE_DEVICES=5 python scripts/train.py task=open_box 


* eval
python scripts/eval.py task=task_name snapshot=path_to_snapshot

python scripts/eval.py task=push_button snapshot=exp_local/20250731/coa_push_button_rlbench_20250731003250/checkpoints/coa_5000.pt num_eval_episodes=1

python scripts/eval_dir.py task=push_button snapshot=exp_local/20250731/coa_push_button_rlbench_20250731003250/checkpoints

bash scripts/eval.sh task=push_button