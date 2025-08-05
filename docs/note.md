分段demo的时候要让每一段数据包含nextbp之后一个step的信息，就是知道下一段应该怎么走


/root/miniforge3/envs/coa/lib/python3.9/site-packages/rlbench/backend/scene.py line234

* data pkl格式保存，读取用pickle.load，类型为rlbench.demo.Demo，获取value方式为demos[i].gripper_open，经过处理后变成converted_demo，每一项都是一个Demostep。low_dim_state相比ee_pos似乎多了一步归一化
- init:
seq_len: 最长子段
load_demos导入demo

* sim
<AppendDemoInfo<ReverseTemporalEnsemble<FrameStack<TimeLimitX<MinMaxNorm<RLBenchEnv instance>>>>>>

init&reset:
Append Info:
ReverseTemporalEnsemble(inherit from ActionSequence): init和reset都初始化action_hist (max_len, action_len, 8)
frame_stack: reset时重复初始画面，传递info给上一层
Time: exec time out
MinMax: norm

step:
Append Info: modify info
ReverseTemporalEnsemble: in _step_sequence() 从长action序列中每次通过对hist加权平均算出一个action交给下一层执行
frame_stack: 
Time: 
MinMax: norm




* model
image_enc: input (View, C, H, W) resnet18+PE
actor_model: 中有三种结构，TransformerEncoderLayer，TransformerDecoderLayer，MTPHeadLayer，后两种区别不大，只有ffn中的线性层数量的区别（后者多一个两层mlp），
encoder将obs处理成mem，decoder从sos token开始，自回归生成之后的动作，每次输出为mtp_size数量的action，decoder中前n-1层都是普通的TransformerDecoderLayer，然后接着相当于是并行经过mtp_size个MTPHeadLayer得到多个action预测

* inference
mtp仅仅在training阶段让模型每个位置的output能预测之后时间步的action，inference只使用第一个mtp layer的输出，输出一个序列(act_len，8)，这个序列从nbp开始反向回归，如果过程中接近当前位置则早停