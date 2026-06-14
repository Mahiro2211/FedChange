# 新服务器部署指南

本指南介绍如何将 FedChange 项目部署到新的 Linux + NVIDIA GPU 服务器上，完成从环境配置到实验运行的全部流程。

---

## 目录布局

```
/home/you/projects/
├── FedChange/          ← 代码（含已迁移的相对路径 JSON）
│   ├── fed_cd/         ← 核心 Python 包
│   ├── partitions/     ← 8 个 Non-IID 划分 JSON（相对路径，跨机器可移植）
│   ├── scripts/        ← 批量实验脚本
│   └── ...
└── WHU-GCD/            ← 数据集
    ├── train/          ← gcd / ugcd_full / ucd / ugcd
    ├── val/            ← 600 样本
    ├── test/           ← 3,300 样本（合成测试集）
    └── test2/          ← 3,906 样本（真实世界测试集）
```

> 如果数据集不在 `../WHU-GCD`，运行时通过 `--data_root /your/path` 指定即可。

---

## 第 1 步：上传文件

```bash
# 在本地机器上打包上传（代码体积小，直接 scp）
scp -r FedChange/ user@server:/home/you/projects/

# 数据集较大，建议用 rsync 断点续传
rsync -avhP WHU-GCD/ user@server:/home/you/projects/WHU-GCD/
```

---

## 第 2 步：创建环境 + 安装依赖

```bash
# SSH 到服务器
ssh user@server

# 创建 conda 环境（推荐 Python 3.10-3.12，不要用 3.13）
conda create -n fedcd python=3.11 -y
conda activate fedcd

# 安装 CUDA 版 PyTorch（必须先装这个，不能用 requirements.txt 里的 CPU 版）
# 根据服务器 CUDA 版本选择 cu118 / cu121 / cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
cd /home/you/projects/FedChange
pip install -r requirements.txt
```

> **重要**：`requirements.txt` 中 `torch>=2.0.0` 会安装 CPU 版，所以**必须先手动安装 CUDA 版 PyTorch**，再安装其他依赖。

---

## 第 3 步：验证 GPU 和数据加载

```bash
cd /home/you/projects/FedChange

# 验证 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 验证数据路径和 partition JSON 能正确加载
python -c "
import json, sys; sys.path.insert(0, '.')
from fed_cd.data.cd_dataset import CDDataset
d = json.load(open('partitions/partition_source.json'))
ds = CDDataset(d['clients']['client_0']['samples'][:3], data_root='../WHU-GCD')
s = ds[0]
print(f'OK: A={s[\"A\"].shape}, L={s[\"L\"].unique().tolist()}')
"
```

期望输出：

```
CUDA: True, GPU: NVIDIA GeForce RTX 4060
OK: A=torch.Size([3, 256, 256]), L=[0, 1]
```

---

## 第 4 步：运行集中式基线

### 手动运行（单次实验）

```bash
cd /home/you/projects/FedChange

# 策略 A: gcd + ugcd_full (3:1)
python -m fed_cd.centralized_main \
    --mode centralized \
    --data_root ../WHU-GCD \
    --train_sources "gcd,ugcd_full" \
    --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 \
    --img_size 256 \
    --epochs 200 \
    --local_bs 8 \
    --lr 0.01 \
    --lr_policy linear \
    --optimizer sgd \
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --save_frequency 20 \
    --project_name "Centralized_A_gcd_ugcdfull" \
    --checkpoint_root results \
    --seed 42

# 策略 B: gcd + ucd + ugcd (6:1:1)
python -m fed_cd.centralized_main \
    --mode centralized \
    --data_root ../WHU-GCD \
    --train_sources "gcd,ucd,ugcd" \
    --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 \
    --img_size 256 \
    --epochs 200 \
    --local_bs 8 \
    --lr 0.01 \
    --lr_policy linear \
    --optimizer sgd \
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --save_frequency 20 \
    --project_name "Centralized_B_gcd_ucd_ugcd" \
    --checkpoint_root results \
    --seed 42
```

### 一键运行

```bash
bash scripts/run_centralized.sh
```

---

## 第 5 步：运行联邦实验

### 手动运行（单个实验）

```bash
cd /home/you/projects/FedChange

# 示例：FedAvg + source partition
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_source.json \
    --data_root ../WHU-GCD \
    --project_name "FedAvg_source_bcd" \
    --fedprox_mu 0.0 \
    --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 \
    --img_size 256 \
    --epochs 200 \
    --frac_num 5 \
    --local_ep 2 \
    --local_bs 8 \
    --lr 0.01 \
    --lr_policy linear \
    --optimizer sgd \
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --save_frequency 20 \
    --checkpoint_root results/fed_bcd \
    --seed 42
```

### 批量运行全部联邦实验（14 个实验 = 2 算法 × 7 partition）

```bash
# 前台运行
bash scripts/run_fed_bcd.sh

# 后台运行 + 日志记录（推荐长时间实验）
nohup bash scripts/run_fed_bcd.sh > fed_bcd_all.log 2>&1 &
tail -f fed_bcd_all.log
```

脚本支持位置参数自定义（顺序：epochs, frac_num, local_ep, batch_size, img_size, lr, fedprox_mu）：

```bash
bash scripts/run_fed_bcd.sh 200 5 2 8 256 0.01 0.01
```

**7 种 partition 对应的 JSON 文件：**

| 名称 | JSON 文件 |
|------|-----------|
| source | `partitions/partition_source.json` |
| dirichlet α=0.1 | `partitions/partition_dirichlet_a0.1_n7.json` |
| dirichlet α=0.5 | `partitions/partition_dirichlet_a0.5_n7.json` |
| dirichlet α=1.0 | `partitions/partition_dirichlet_a1.0_n7.json` |
| IID (α=100) | `partitions/partition_dirichlet_a100.0_n7.json` |
| class c1 | `partitions/partition_class_c1_separate.json` |
| hybrid | `partitions/partition_hybrid_c1_separate.json` |

**2 种联邦算法：**

| 算法 | 参数 |
|------|------|
| FedAvg | `--fedprox_mu 0.0` |
| FedProx | `--fedprox_mu 0.01`（或其他正值） |

---

## 第 6 步（可选）：重新生成 Non-IID 划分

如果需要用不同的客户端数量或 Dirichlet 参数重新划分：

```bash
cd /home/you/projects/FedChange

# Dirichlet α=0.3, 10 个客户端
python partition_dirichlet.py --data_root ../WHU-GCD --alpha 0.3 --num_clients 10

# 按来源划分
python partition_by_source.py --data_root ../WHU-GCD --clients_per_source 3,3,2,2

# 按类别划分
python partition_by_class.py --data_root ../WHU-GCD --strategy separate --class_id 1

# 混合策略
python partition_hybrid.py --data_root ../WHU-GCD --source_strategy separate --class_id 1
```

新生成的 JSON 同样使用相对路径，跨机器可移植。

---

## 第 7 步（可选）：路径迁移工具

如果 partition JSON 中包含旧的绝对路径（从其他机器复制的），用以下命令一键转换为相对路径：

```bash
# 检查（dry-run，不修改文件）
python migrate_paths.py --check

# 执行迁移
python migrate_paths.py

# 迁移单个文件
python migrate_paths.py --file partitions/partition_source.json
```

---

## 实验矩阵总览

### 集中式基线（2 个实验）

| 实验名称 | 训练数据 | 命令 |
|----------|---------|------|
| Centralized_A | gcd + ugcd_full (3:1) | `bash scripts/run_centralized.sh` |
| Centralized_B | gcd + ucd + ugcd (6:1:1) | 同上（脚本内含两条） |

### 联邦实验（14 个实验）

| | source | dir α=0.1 | dir α=0.5 | dir α=1.0 | IID | class c1 | hybrid |
|---|---|---|---|---|---|---|---|
| **FedAvg** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **FedProx** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

一键运行：`bash scripts/run_fed_bcd.sh`

---

## 常用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `../WHU-GCD` | 数据集根目录（**迁移时只需改这一个**） |
| `--epochs` | 200 | 全局轮数（联邦）或 epoch 数（集中式） |
| `--frac_num` | 5 | 每轮参与的客户端数 |
| `--local_ep` | 2 | 本地训练 epoch 数 |
| `--local_bs` | 8 | 本地训练 batch size |
| `--lr` | 0.01 | 学习率 |
| `--fedprox_mu` | 0.0 | FedProx 近端项权重（0 = FedAvg） |
| `--img_size` | 256 | 图像裁剪尺寸 |
| `--num_classes` | 2 | 2 = BCD，8 = SCD |
| `--pretrained` | True | 使用 ImageNet 预训练 ResNet 权重 |
| `--eval_splits` | val,test,test2 | 评估的数据集 |
| `--global_test_frequency` | 20 | 每 N 轮评估一次 |
| `--save_frequency` | 20 | 每 N 轮保存一次 checkpoint |
| `--seed` | 42 | 随机种子 |

---

## 预计耗时

| 步骤 | 耗时（单 GPU） |
|------|---------------|
| 环境安装 | ~5 分钟 |
| 验证 | ~1 分钟 |
| 集中式基线 ×2 | ~数小时 |
| 联邦实验 ×14 | ~数天 |
| 重新生成划分 | ~2 分钟/种 |

> 具体耗时取决于 GPU 型号和数据加载速度。

---

## 常见问题

### Q: `CUDA: False`？

PyTorch 未检测到 GPU。检查：

1. `nvidia-smi` 确认 GPU 驱动正常
2. 确认安装了 CUDA 版 PyTorch：`pip show torch` 查看版本号是否含 `+cu121`
3. 如果误装了 CPU 版，卸载重装：`pip uninstall torch torchvision && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Q: `FileNotFoundError: train/gcd/im1/xxx.png`？

`--data_root` 路径不正确。检查：

```bash
ls ../WHU-GCD/train/gcd/im1/ | head
```

如果路径不对，用 `--data_root /correct/path/to/WHU-GCD` 指定。

### Q: OOM (Out of Memory)？

减小 batch size：`--local_bs 4` 或 `--local_bs 2`。

### Q: 如何恢复中断的实验？

Checkpoint 保存在 `results/<project_name>/` 下。查看是否有 `latest.pth` 或 `round_NN.pth`，手动加载或修改代码从 checkpoint 恢复。

### Q: 如何用 tmux 长时间运行？

```bash
tmux new -s fedcd
conda activate fedcd
cd /home/you/projects/FedChange
nohup bash scripts/run_fed_bcd.sh > fed_bcd_all.log 2>&1 &
# Ctrl+B, D 脱离 tmux
# tmux attach -t fedcd 重新连接
```
