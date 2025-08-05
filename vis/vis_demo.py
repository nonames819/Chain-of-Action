import pickle, os
import numpy as np
import imageio

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

def visualize_from_pickle(pickle_file, task, index=0):
    pkl_path = f"/workspace/chd_data/Chain-of-Action/data/rlbench/eval/{task}/variation0/episodes/episode{index}/low_dim_obs.pkl"
    output_dir = f"/workspace/chd_data/Chain-of-Action/vis/{task}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"episode{index}.mp4")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    frames = []
    
    for i in range(len(data)):
        # 从 data[i] 中提取每个视角图像
        obs = data[i]
        img1 = obs.left_shoulder_rgb
        img2 = obs.right_shoulder_rgb
        img3 = obs.wrist_rgb
        img4 = obs.front_rgb

        # 确保所有图像大小一致
        assert img1.shape == img2.shape == img3.shape == img4.shape == (128, 128, 3), f"Shape mismatch at frame {i}"

        # 2x2 拼接图像
        top = np.hstack((img1, img2))
        bottom = np.hstack((img3, img4))
        combined = np.vstack((top, bottom))

        frames.append(combined)

        # print(f"Frame {i}: combined shape {combined.shape}")

    # 保存为视频（imageio 自动处理帧率与编码器）
    imageio.mimwrite(output_file, frames, fps=0, codec='libx264')

    print(f"Video saved to: {output_file}")

def main():
    for task in tasks:
        visualize_from_pickle(pickle_file=None, task=task, index=0)

if __name__ == "__main__":
    main()
    
