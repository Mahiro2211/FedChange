# 新服务器部署指南

本指南介绍如何将 FedChange 项目部署到新的 Linux + NVIDIA GPU 服务器上，完成从环境配置到实验运行的全部流程。

---

## 目录布局

```
/home/you/projects/
├── FedChange/          ← 代码（含已迁移的相对路径 JSON）
│   ├── fed_cd/         ← 核心 Python 包
│   ├── partitions/     ← Dirichlet 划分包（generate.py + 4 个 JSON，相对路径可移植）
│   ├── scripts/        ← 批量实验脚本（Bash）
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
d = json.load(open('partitions/partition_dirichlet_a0.5_n7.json'))
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

> 所有运行脚本启动时会**自动检查依赖**：缺核心库或缺 torchange 基线库（仅 torchange_fed / changen2 步骤需要）时，会打印中文提示与安装命令并中止。也可手动检查：`source scripts/check_env.sh && check_env_core && check_env_torchange`。

---

## 第 4 步：运行联邦实验

### 手动运行（单个实验）

```bash
cd /home/you/projects/FedChange

# 示例：FedAvg + Dirichlet α=0.5 partition
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a0.5_n7.json \
    --data_root ../WHU-GCD \
    --project_name "FedAvg_dirichlet05_bcd" \
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

### 批量运行全部联邦实验（8 个实验 = 2 算法 × 4 Dirichlet α）

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

**4 档 Dirichlet α 对应的 JSON 文件：**

| 名称 | JSON 文件 |
|------|-----------|
| dirichlet α=0.1 | `partitions/partition_dirichlet_a0.1_n7.json` |
| dirichlet α=0.5 | `partitions/partition_dirichlet_a0.5_n7.json` |
| dirichlet α=1.0 | `partitions/partition_dirichlet_a1.0_n7.json` |
| 近 IID (α=100) | `partitions/partition_dirichlet_a100.0_n7.json` |

> 缺失时用 `python -m partitions.generate --alpha <α> --num_clients 7 --data_root ../WHU-GCD` 重新生成。

### 联邦算法详解（4 种可组合算法）

本框架的联邦算法由 **3 个独立开关**控制，可自由组合，共覆盖 4 类算法：

| 算法 | `--iid` | `--fedprox_mu` | `--globalema` | 原理 |
|------|---------|----------------|---------------|------|
| **FedAvg**（简单平均） | `True` | `0.0` | `False` | 选中客户端权重简单平均 |
| **加权 FedAvg** | `False` | `0.0` | `False` | 按客户端数据量加权平均（**Non-IID 默认**） |
| **FedProx** | `False` | `>0`（如 0.01） | `False` | 加权平均 + 本地近端正则 `μ/2·‖w−w_t‖²` |
| **全局 EMA**（可叠加） | 任意 | 任意 | `True` | 对全局模型参数做指数移动平均（decay 0.999） |

> `--iid` 仅决定**聚合方式**：`True`=简单平均，`False`=按数据量加权。`--fedprox_mu` 决定**本地训练**是否加近端约束。`--globalema` 决定聚合后是否做 EMA 平滑。三者正交，可任意叠加。

#### 各算法运行命令

**1. 纯 FedAvg（IID 聚合）**
```bash
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a100.0_n7.json \
    --data_root ../WHU-GCD --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 --img_size 256 --epochs 200 --frac_num 5 \
    --local_ep 2 --local_bs 8 --lr 0.01 --lr_policy linear \
    --pretrained True --iid True --fedprox_mu 0.0 \
    --eval_splits "val,test,test2" \
    --project_name "FedAvg_iid_bcd" --checkpoint_root results/fed_bcd
```

**2. 加权 FedAvg（Non-IID 聚合，推荐用于异构划分）**
```bash
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a0.5_n7.json \
    --data_root ../WHU-GCD --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 --img_size 256 --epochs 200 --frac_num 5 \
    --local_ep 2 --local_bs 8 --lr 0.01 --lr_policy linear \
    --pretrained True --iid False --fedprox_mu 0.0 \
    --eval_splits "val,test,test2" \
    --project_name "FedAvg_dirichlet05_bcd" --checkpoint_root results/fed_bcd
```

**3. FedProx（加权聚合 + 近端约束）**
```bash
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a0.5_n7.json \
    --data_root ../WHU-GCD --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 --img_size 256 --epochs 200 --frac_num 5 \
    --local_ep 2 --local_bs 8 --lr 0.01 --lr_policy linear \
    --pretrained True --iid False --fedprox_mu 0.01 \
    --eval_splits "val,test,test2" \
    --project_name "FedProx0.01_dirichlet05_bcd" --checkpoint_root results/fed_bcd
```

**4. FedProx + 全局 EMA（组合算法）**
```bash
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a0.5_n7.json \
    --data_root ../WHU-GCD --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 --img_size 256 --epochs 200 --frac_num 5 \
    --local_ep 2 --local_bs 8 --lr 0.01 --lr_policy linear \
    --pretrained True --iid False --fedprox_mu 0.01 --globalema True \
    --eval_splits "val,test,test2" \
    --project_name "FedProx0.01_EMA_dirichlet05_bcd" --checkpoint_root results/fed_bcd
```

> **FedProx μ 取值建议**：异构越强 μ 越大。常用 `0.001 / 0.01 / 0.1 / 1.0` 扫描；μ=0 即退化为加权 FedAvg。

### 用不同骨干/模型跑联邦实验

联邦聚合基于 `state_dict`，**与模型结构无关**，因此任何 `--net_G` 都能直接联邦化（torchange 模型由适配器自动处理原生损失）：

```bash
# 用 ChangeSparseBCD 跑 FedAvg（与 BIT-CD 同划分、同调度对比）
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a0.5_n7.json \
    --data_root ../WHU-GCD --net_G changesparse_bcd \
    --num_classes 2 --img_size 256 --epochs 200 --frac_num 5 \
    --local_ep 2 --local_bs 8 --lr 0.01 --lr_policy linear \
    --pretrained True --iid False --fedprox_mu 0.0 \
    --eval_splits "val,test,test2" \
    --project_name "FedAvg_changesparse_bcd_dirichlet05" \
    --checkpoint_root results/torchange_fed
```

一键批量对比所有 torchange 基线 × (FedAvg + FedProx)：

```bash
bash scripts/run_fed_torchange_bcd.sh
```

> torchange 模型每轮本地训练用其**原生 BCE+Dice 损失**（BIT-CD 用 CrossEntropyLoss）；评估统一走 2 类 logits + argmax，指标口径完全一致。

### 使用 torchange 基线（可选安装）

torchange 基线是可选依赖，**仅当 `--net_G` 选 torchange 模型时才需要**。BIT-CD 实验（`base_*`）无需安装。

```bash
conda activate fedcd
pip install torchange "albumentations>=2.0.0" tifffile scikit-image \
    datasets ever-beta segmentation-models-pytorch timm
```

验证（应输出 `Eval logits: (1, 2, 256, 256)` 与一个 loss 标量）：

```bash
cd /home/you/projects/FedChange
python -c "
from fed_cd.models import build_cd_model
import torch
m = build_cd_model('changesparse_bcd', pretrained=False)
x1, x2 = torch.randn(1,3,256,256), torch.randn(1,3,256,256)
print('Eval logits:', m(x1, x2).shape)
print('Train loss:', float(m.compute_loss(x1, x2, torch.zeros(1,256,256).long())))
"
```

**Changen2 零样本评估**（免训练，自动从 HuggingFace 下载预训练权重）：

```bash
python scripts/run_changen2_zeroshot.py --data_root ../WHU-GCD
```

> 首次运行 torchange 模型若 `--pretrained True`，会自动下载 ImageNet 预训练骨干权重（ResNet/Swin）；Changen2 另需从 `EVER-Z/Changen2-ChangeStar1x256` 下载。确保服务器可访问 `download.pytorch.org` 与 `huggingface.co`。

---

## 第 5 步（可选）：重新生成 Dirichlet 划分

如果需要用不同的客户端数量或 Dirichlet α 重新划分：

```bash
cd /home/you/projects/FedChange

# Dirichlet α=0.3, 10 个客户端
python -m partitions.generate --data_root ../WHU-GCD --alpha 0.3 --num_clients 10

# 预览统计不写文件
python -m partitions.generate --alpha 0.5 --dry_run

# 排除负样本（ucd/ugcd/ugcd_full）
python -m partitions.generate --alpha 0.5 --no_neg
```

新生成的 JSON 同样使用相对路径，跨机器可移植。命名规则 `partition_dirichlet_a{α}_n{客户端数}.json`。

---

## 实验矩阵总览

### 集中式上界 + 近 IID 参考

| 实验 | 模型 (`--net_G`) | 划分 | 命令 |
|------|------------------|------|------|
| **BIT-CD 集中式（上界）** | `base_transformer_pos_s4_dd8` | 全部数据集中训练（非联邦） | `bash scripts/run_centralized.sh` |
| torchange 集中式 ×4（可选） | `changesparse_bcd`、`changestar_1xd_r18`、`changestar_1xd`、`changestar_2_5` | 全部数据集中训练 | `bash scripts/run_centralized.sh 200 8 256 0.01 <net_G> <tag>` |
| BIT-CD 近 IID（参考） | `base_transformer_pos_s4_dd8` | `partition_dirichlet_a100.0_n7.json` (`--iid True`) | 见第 4 步 |
| torchange 近 IID ×4 | `changesparse_bcd`、`changestar_1xd_r18`、`changestar_1xd`、`changestar_2_5` | `partition_dirichlet_a100.0_n7.json` | `bash scripts/run_fed_torchange_bcd.sh`（改指向近 IID 划分） |
| Changen2 零样本 | `changen2_zeroshot` | —（免训练） | `python scripts/run_changen2_zeroshot.py` |

> 性能上界由**集中式训练**（`fed_cd.centralized.cen_main`）提供；α=100 的联邦训练作为近 IID 参考。集中式与联邦共享同一训练集、模型和超参，唯一区别是优化方式。

### 联邦实验（Non-IID 鲁棒性）

**主矩阵：BIT-CD × 2 算法 × 4 Dirichlet α = 8 个实验**

| | dir α=0.1 | dir α=0.5 | dir α=1.0 | dir α=100 |
|---|---|---|---|---|
| **FedAvg**（加权） | ✓ | ✓ | ✓ | ✓ |
| **FedProx** μ=0.01 | ✓ | ✓ | ✓ | ✓ |

一键运行：`bash scripts/run_fed_bcd.sh`

**跨模型联邦对比：torchange ×4 × 2 算法（默认 dir α=0.5）**

| | changesparse_bcd | changestar_1xd_r18 | changestar_1xd | changestar_2_5 |
|---|---|---|---|---|
| **FedAvg** | ✓ | ✓ | ✓ | ✓ |
| **FedProx** μ=0.01 | ✓ | ✓ | ✓ | ✓ |

一键运行：`bash scripts/run_fed_torchange_bcd.sh`

> 扩展到更多 α：取消 `scripts/run_fed_torchange_bcd.sh` 内 `PARTITIONS` 数组的注释行（α=0.1 / α=100）。

**算法变体（任意模型/划分可叠加）**：把 `--fedprox_mu 0.0 --globalema False --iid True` 作为基准，按需开启 `--iid False`（加权）、`--fedprox_mu 0.01`（近端）、`--globalema True`（EMA）。

### 联邦算法对比（4 算法）

固定 Dirichlet α=0.5，对比 FedAvg / FedProx / FedNova / SCAFFOLD：

一键运行：`bash scripts/run_fed_alg_comparison.sh`

---

## 常用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `../WHU-GCD` | 数据集根目录（**迁移时只需改这一个**） |
| `--net_G` | `base_transformer_pos_s4_dd8` | 模型/骨干：BIT-CD 变体（`base_*`）或 torchange 基线（`changesparse_bcd` / `changestar_1xd` / `changestar_1xd_r18` / `changestar_2_5` / `changen2_zeroshot`） |
| `--num_classes` | 2 | 2 = BCD（本框架为二值变化检测） |
| `--epochs` | 200 | 全局通信轮数 |
| `--frac_num` | 5 | 每轮参与的客户端数 |
| `--local_ep` | 2 | 本地训练 epoch 数 |
| `--local_bs` | 8 | 本地训练 batch size |
| `--lr` | 0.01 | 学习率 |
| `--lr_policy` | linear | 学习率衰减（linear / step） |
| `--optimizer` | sgd | sgd（momentum=0.9）或 adam |
| `--iid` | False | **聚合方式**：True=简单平均，False=按数据量加权（Non-IID 默认） |
| `--fedprox_mu` | 0.0 | FedProx 近端项权重（0 = 无近端约束） |
| `--globalema` | False | 聚合后对全局模型做 EMA 平滑（decay 0.999） |
| `--img_size` | 256 | 图像裁剪尺寸 |
| `--pretrained` | True | 使用 ImageNet 预训练骨干权重 |
| `--eval_splits` | val,test,test2 | 评估的数据集 |
| `--global_test_frequency` | 20 | 每 N 轮评估一次 |
| `--save_frequency` | 20 | 每 N 轮保存一次 checkpoint |
| `--seed` | 42 | 随机种子 |

> **算法组合速查**：`--iid` 控聚合、`--fedprox_mu` 控本地近端、`--globalema` 控 EMA，三者正交。例如 `--iid False --fedprox_mu 0.01 --globalema True` = 加权聚合 + FedProx + 全局 EMA。

---

## 预计耗时

| 步骤 | 耗时（单 GPU） |
|------|---------------|
| 环境安装（含 torchange 可选） | ~5–10 分钟 |
| 验证 | ~1 分钟 |
| 联邦实验 ×8（BIT-CD，4 α × 2 算法） | ~数天 |
| 联邦 torchange 对比 ×8 | ~数天 |
| Changen2 零样本评估 | ~数分钟（含首次权重下载） |
| 重新生成划分 | ~2 分钟/种 |

> 具体耗时取决于 GPU 型号、骨干大小（R18 < R50 < ViT-B）和数据加载速度。大骨干模型 OOM 时减小 `--local_bs`。

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

### Q: 如何选择骨干网络？

- **公平受控对比**（隔离变化建模贡献）：选 ResNet18 系列——BIT-CD、`changesparse_bcd`、`changestar_1xd_r18`。
- **模型容量上界**（看最强表现）：选大骨干——`changestar_1xd`（R50）、`changestar_2_5`（R50）、`changen2_zeroshot`（ViT-B）。
- **仅验证 BIT-CD 联邦效果**：用默认 `base_transformer_pos_s4_dd8` 即可，无需装 torchange。

> torchange 模型需先安装 torchange（见第 4 步"使用 torchange 基线（可选安装）"）。BIT-CD 变体无需任何额外依赖。

### Q: 如何选择/组合联邦算法？

三个正交开关，按异构程度递增推荐：

| 场景 | 推荐配置 |
|------|---------|
| IID / 接近 IID | `--iid True --fedprox_mu 0.0`（纯 FedAvg） |
| 中度 Non-IID | `--iid False --fedprox_mu 0.0`（加权 FedAvg，**默认**） |
| 极端 Non-IID | `--iid False --fedprox_mu 0.01`（FedProx） |
| 训练不稳定/震荡 | 上述任意 + `--globalema True`（加 EMA 平滑） |

> 异构越强（Dirichlet α 越小），FedProx 的 μ 可适当调大（0.01→0.1）。先跑加权 FedAvg 作基线，再叠加近端/EMA 看增益。
