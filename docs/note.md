分段demo的时候要让每一段数据包含nextbp之后一个step的信息，就是知道下一段应该怎么走


/root/miniforge3/envs/coa/lib/python3.9/site-packages/rlbench/backend/scene.py line234

* data
- init:
seq_len: 最长子段


* model
image_enc: input (View, C, H, W) resnet18+PE
