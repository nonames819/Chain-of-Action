import os
import pickle
import json
import numpy as np
from typing import Any, Dict, List

tasks = [ 
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
] 

def extract_gripper_data(task, index=0):
    """
    从pickle文件中提取gripper相关数据：gripper_open, gripper_pose, gripper_matrix
    """
    pkl_path = f"/workspace/chd_data/Chain-of-Action/data/rlbench/eval/{task}/variation0/episodes/episode{index}/low_dim_obs.pkl"
    output_dir = f"/workspace/chd_data/Chain-of-Action/vis/{task}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"eval_episode{index}_gripper_data.txt")
    
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"Processing {task}, episode {index}...")
        print(f"Total frames: {len(data)}")
        
        # 需要提取的gripper相关字段
        target_fields = ['gripper_open', 'gripper_pose', 'gripper_matrix']
        
        # 存储gripper数据
        gripper_data = {
            "task": task,
            "episode": index,
            "total_frames": len(data),
            "gripper_open": [],
            "gripper_pose": [],
            "gripper_matrix": []
        }
        
        # 循环读取每个时间步的数据
        for i in range(len(data)):
            obs = data[i]
            
            # 提取gripper_open
            if hasattr(obs, 'gripper_open'):
                gripper_open = obs.gripper_open
                if isinstance(gripper_open, np.ndarray):
                    gripper_data["gripper_open"].append(gripper_open.tolist())
                else:
                    gripper_data["gripper_open"].append(gripper_open)
            else:
                gripper_data["gripper_open"].append(None)
            
            # 提取gripper_pose
            if hasattr(obs, 'gripper_pose'):
                gripper_pose = obs.gripper_pose
                if isinstance(gripper_pose, np.ndarray):
                    gripper_data["gripper_pose"].append(gripper_pose.tolist())
                else:
                    gripper_data["gripper_pose"].append(gripper_pose)
            else:
                gripper_data["gripper_pose"].append(None)
            
            # 提取gripper_matrix
            if hasattr(obs, 'gripper_matrix'):
                gripper_matrix = obs.gripper_matrix
                if isinstance(gripper_matrix, np.ndarray):
                    gripper_data["gripper_matrix"].append(gripper_matrix.tolist())
                else:
                    gripper_data["gripper_matrix"].append(gripper_matrix)
            else:
                gripper_data["gripper_matrix"].append(None)
        
        with open(output_file, 'w') as f:
            f.write(f"Task: {task}\n")
            f.write(f"Episode: {index}\n")
            f.write(f"Total Frames: {len(data)}\n")
            f.write("="*80 + "\n\n")
            
            for i in range(len(data)):
                f.write(f"Frame {i:3d}:\n")
                
                # gripper_open
                gripper_open = gripper_data["gripper_open"][i]
                if gripper_open is not None:
                    if isinstance(gripper_open, list):
                        gripper_open_str = ", ".join([f"{x:.4f}" for x in gripper_open])
                    else:
                        gripper_open_str = f"{gripper_open:.4f}"
                    f.write(f"  gripper_open:   {gripper_open_str}\n")
                else:
                    f.write(f"  gripper_open:   None\n")
                
                # gripper_pose (保持在一行)
                gripper_pose = gripper_data["gripper_pose"][i]
                if gripper_pose is not None:
                    pose_str = "[" + ", ".join([f"{x:.4f}" for x in gripper_pose]) + "]"
                    f.write(f"  gripper_pose:   {pose_str}\n")
                else:
                    f.write(f"  gripper_pose:   None\n")
                
                # gripper_matrix (4x4矩阵格式)
                gripper_matrix = gripper_data["gripper_matrix"][i]
                if gripper_matrix is not None:
                    f.write(f"  gripper_matrix:\n")
                    matrix = np.array(gripper_matrix)
                    if matrix.shape == (4, 4):
                        for row in range(4):
                            row_str = "    [" + ", ".join([f"{matrix[row, col]:8.4f}" for col in range(4)]) + "]"
                            f.write(f"{row_str}\n")
                    else:
                        # 如果不是4x4矩阵，按原格式输出
                        f.write(f"    {matrix.tolist()}\n")
                else:
                    f.write(f"  gripper_matrix: None\n")
                
                f.write("\n")
        
        print(f"Gripper data saved to: {output_file}")
        
        # 打印数据概览
        print(f"Data shapes:")
        print(f"  gripper_open: {len(gripper_data['gripper_open'])} timesteps")
        print(f"  gripper_pose: {len(gripper_data['gripper_pose'])} timesteps")
        print(f"  gripper_matrix: {len(gripper_data['gripper_matrix'])} timesteps")
        
        # 打印第一个时间步的数据样例
        if gripper_data["gripper_open"]:
            print(f"Sample data (first timestep):")
            print(f"  gripper_open: {gripper_data['gripper_open'][0]}")
            if gripper_data['gripper_pose'][0] is not None:
                pose_shape = np.array(gripper_data['gripper_pose'][0]).shape if gripper_data['gripper_pose'][0] is not None else "None"
                print(f"  gripper_pose shape: {pose_shape}")
            if gripper_data['gripper_matrix'][0] is not None:
                matrix_shape = np.array(gripper_data['gripper_matrix'][0]).shape if gripper_data['gripper_matrix'][0] is not None else "None"
                print(f"  gripper_matrix shape: {matrix_shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File not found - {pkl_path}")
        return False
    except Exception as e:
        print(f"Error processing {task}: {e}")
        return False

def main():
    """
    主函数：处理所有任务，提取gripper相关数据
    """
    print("Starting gripper data extraction...")
    
    successful_tasks = []
    failed_tasks = []
    
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Processing task: {task}")
        print(f"{'='*50}")
        
        # 提取gripper数据
        success = extract_gripper_data(task, index=0)
        
        if success:
            successful_tasks.append(task)
        else:
            failed_tasks.append(task)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    
    if successful_tasks:
        print(f"\nSuccessful tasks: {', '.join(successful_tasks)}")
    
    if failed_tasks:
        print(f"\nFailed tasks: {', '.join(failed_tasks)}")
    
    print(f"\nGripper data saved to: /workspace/chd_data/Chain-of-Action/vis/[task_name]/episode0_gripper_data.txt")

if __name__ == "__main__":
    main()