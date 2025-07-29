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
python scripts/train.py task=task_name 
python scripts/train.py task=push_button 


* eval
python scripts/eval.py task=task_name snapshot=path_to_snapshot

python scripts/eval.py task=push_button snapshot=/workspace/chd_data/Chain-of-Action/ckpt/official/rlbench_push_button_20250702092806/checkpoints/rlbench_push_button_20250702092806_coa_20000.pt

bash scripts/eval.sh task=push_button