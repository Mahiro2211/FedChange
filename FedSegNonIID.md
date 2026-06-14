FedSeg 的 Non-IID 数据划分方式
论文：*"FedSeg: Class-Heterogeneous Federated Learning for Semantic Segmentation"* (CVPR 2023, Miao et al.)
划分方法：类别异构（Class-Heterogeneous），而非 Dirichlet 分布
FedSeg 不使用 Dirichlet 分布划分，而是采用一种极端的 按类别划分 方式：
1. 预处理（"erase" 机制）：先将数据集按语义类别拆分到独立的子文件夹中。例如 Cityscapes 的 19 个类别各自形成一个文件夹，每个文件夹只包含该类别相关的图像。
2. 客户端分配（cityscapes_noniid_extend 函数）：
- 每个类别均匀分配给固定数量的客户端（num_users_per_class = num_users / num_classes）
- 每个类别文件夹内的图像随机等分给该类别对应的客户端
- 每个客户端只拥有一个类别的数据 → 极端 Non-IID
3. 每轮训练采样 frac_num=5 个客户端参与。
客户端数量（随数据集不同而变化）
数据集	类别数	每类客户端数
Cityscapes	19	8
ADE20K	150	3
Pascal VOC	20	3
CamVid	11	2
公式：总客户端数 = 类别数 × 每类客户端数
代码中的关键逻辑（datasplit.py）
client_index = i + class_idx × num_users_per_class
即：客户端 0~7 → 类别 0，客户端 8~15 → 类别 1，依此类推。每个客户端的数据完全来自单一类别，形成严格的类别异构 Non-IID 场景。
聚合时对 non-IID 使用 weighted_average_weights（按客户端数据量加权），而非简单平均。