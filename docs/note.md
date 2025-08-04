分段demo的时候要让每一段数据包含nextbp之后一个step的信息，就是知道下一段应该怎么走


/root/miniforge3/envs/coa/lib/python3.9/site-packages/rlbench/backend/scene.py line234

* data
- init:
seq_len: 最长子段


* model
image_enc: input (View, C, H, W) resnet18+PE
actor_model: 中有三种结构，TransformerEncoderLayer，TransformerDecoderLayer，MTPHeadLayer，后两种区别不大，只有ffn中的线性层数量的区别（后者多一个两层mlp），
encoder将obs处理成mem，decoder从sos token开始，自回归生成之后的动作，每次输出为mtp_size数量的action，decoder中前n-1层都是普通的TransformerDecoderLayer，然后接着相当于是并行经过mtp_size个MTPHeadLayer得到多个action预测