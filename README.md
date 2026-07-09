# FedChange: 联邦学习遥感图像变化检测

<p align="center">
  <strong>Federated Learning for Remote Sensing Image Change Detection on WHU-GCD</strong>
</p>

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据集说明](#数据集说明)
- [Non-IID 数据划分](#non-iid-数据划分)
- [快速开始](#快速开始)
- [联邦学习训练](#联邦学习训练)
- [torchange 对比基线](#torchange-对比基线)
- [评估与结果汇总](#评估与结果汇总)
- [技术架构](#技术架构)
- [实验设计](#实验设计)
- [引用](#引用)

---

## 项目概述

FedChange 是一个将**联邦学习（Federated Learning）**应用于**遥感图像变化检测（Change Detection）**的完整实验框架。它基于以下两个核心工作构建：

| 组件 | 来源 | 论文 |
|------|------|------|
| **BIT-CD** | 变化检测模型 | Chen et al., *"Remote Sensing Image Change Detection with Transformers"*, IEEE TGRS, 2021 |
| **FedSeg** | 联邦学习框架 | Miao et al., *"FedSeg: Class-Heterogeneous Federated Learning for Semantic Segmentation"*, CVPR, 2023 |

项目在 **WHU-GCD**（武汉大学生成式变化检测数据集）上验证联邦学习的可行性，核心研究问题是：**在数据分布异构（Non-IID）的多客户端场景下，能否有效地训练变化检测模型？**

---

## 核心特性

- **模型无关的联邦框架**：权重聚合与模型结构完全解耦，可适配任意 PyTorch 模型
- **4 种 SOTA 联邦算法**：FedAvg、FedProx、FedNova、SCAFFOLD，通过统一参数 `--fed_alg` 路由（详见 [联邦算法](#联邦算法)）
- **Dirichlet Non-IID 数据划分**：用 Dirichlet 分布的浓度参数 α 控制客户端间异构度（α 越小越异构，α→∞ 趋近 IID），扫描多档 α 覆盖从极端异构到近 IID 的连续谱
- **二值变化检测（BCD）**：聚焦二值变化检测任务（2 类：无变化/变化像素级分割）
- **BIT-CD Transformer 模型**：自包含实现，基于 ResNet18 + Bitemporal Transformer，12.4M 参数
- **torchange 对比基线**：集成 [torchange](https://github.com/Z-Zheng/pytorch-change-models) 库的 4 个 SOTA 变化检测模型 + 1 个零样本基础模型，支持联邦对比（详见 [torchange 对比基线](#torchange 对比基线)）
- **多随机种子重复实验**：所有批量脚本支持多 seed 循环，`summarize.py` 自动按 `_s<seed>` 分组并产出 mean±std 表
- **可视化工具**：`fed_cd/plot.py` 一键生成收敛曲线、训练 loss 曲线、Dirichlet α 扫描（带 errorbar）、预测 TP/FP/FN 着色图
- **完整的评估体系**：F1、IoU、Precision、Recall，在合成测试集和真实测试集上分别评估
- **集中式训练上界**：提供集中式 BIT-CD 训练（`fed_cd.centralized.cen_main`）作为性能上界，与联邦实验同框架、同超参对比
- **灵活的运行方式**：`run_all.sh` 一键跑全部实验，也支持 `--only/--skip` 选择性运行单个对比试验（详见 [运行全部对比实验](#运行全部对比实验)）

---

## 项目结构

```
FedChange/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
│
├── partitions/                        # Dirichlet 划分包（Python 包 + 生成的 JSON）
│   ├── __init__.py
│   ├── generate.py                    #   Dirichlet 划分生成器（扫描、划分、统计、CLI）
│   ├── partition_dirichlet_a0.1_n7.json     # α=0.1 极端异构（gitignored，按需重生）
│   ├── partition_dirichlet_a0.5_n7.json     # α=0.5 中等异构
│   ├── partition_dirichlet_a1.0_n7.json     # α=1.0 轻度异构
│   └── partition_dirichlet_a100.0_n7.json   # α=100 近 IID
│
├── fed_cd/                            # 核心代码包
│   ├── options.py                     # 统一命令行参数管理
│   ├── summarize.py                   # 实验结果汇总工具
│   │
│   ├── data/                          # 数据处理
│   │   ├── cd_dataset.py              #   WHU-GCD 双时相数据集（BCD）
│   │   ├── data_partition.py          #   划分加载 + 评估集扫描
│   │   └── data_utils.py              #   同步双时相数据增强
│   │
│   ├── models/                        # 模型
│   │   ├── bit_cd.py                  #   BIT-CD 模型（ResNet18 + Transformer）
│   │   ├── torchange_adapter.py       #   torchange 基线适配器（统一 forward/compute_loss 接口）
│   │   └── __init__.py                #   统一模型工厂 build_cd_model()
│   │
│   ├── federated/                     # 联邦学习（FedAvg/FedProx/FedNova/SCAFFOLD）
│   │   ├── fed_main.py                #   联邦主循环（算法路由 + c_global 维护）
│   │   ├── local_update.py            #   本地训练（含 FedProx/SCAFFOLD control variate）
│   │   └── aggregation.py             #   权重聚合（FedAvg/FedNova/SCAFFOLD）+ EMA
│   ├── centralized/                   # 集中式训练（性能上界）
│   │   └── cen_main.py                #   集中式主循环（与联邦同框架对比）
│   │
│   ├── evaluation/                    # 评估
│   │   ├── cd_metrics.py              #   混淆矩阵指标计算
│   │   └── evaluator.py               #   多集评估器
│   └── plot.py                        # 可视化（收敛曲线/loss/α扫描/预测着色）
│
│
├── scripts/                           # 运行脚本（Bash，用 Git Bash / WSL / Linux 运行）
│   ├── run_all.sh                     # ⭐ 总入口：单 seed 跑全部 5 类实验，支持 --only/--skip
│   ├── check_env.sh                   # 共享依赖检查（核心库 + torchange 可选库）
│   ├── run_fed_alg_comparison.sh      # 4 联邦算法对比（FedAvg/FedProx/FedNova/SCAFFOLD）
│   ├── run_fed_bcd.sh                 # BCD 联邦实验（4 个 Dirichlet α × FedAvg+FedProx × 多 seed）
│   ├── run_fed_torchange_bcd.sh       # torchange 联邦对比（多 seed）
│   ├── run_centralized.sh             # 集中式训练上界
│   └── run_changen2_zeroshot.py       # Changen2 零样本评估（免训练）
│
└── results/                           # 实验输出（训练自动创建）
```

---

## 环境配置

### 系统要求

- Python ≥ 3.10
- PyTorch ≥ 2.0（推荐 GPU 版本）
- NVIDIA GPU（推荐 ≥ 8GB 显存）

### 安装依赖

```bash
cd FedChange
pip install -r requirements.txt
```

### 安装 torchange 对比基线（可选）

若要运行 [torchange 对比基线](#torchange-对比基线)，需额外安装 torchange 及其依赖：

```bash
# 稳定版（PyPI）
pip install torchange

# 或最新版（GitHub master）
pip install -U --no-deps --force-reinstall git+https://github.com/Z-Zheng/pytorch-change-models

# torchange 运行所需依赖（--no-deps 安装时需手动补齐）
pip install "albumentations>=2.0.0" tifffile scikit-image datasets ever-beta segmentation-models-pytorch
```

验证安装：

```bash
python -c "
from fed_cd.models import build_cd_model
import torch
m = build_cd_model('changesparse_bcd', pretrained=False)
x1, x2 = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
print('Eval logits:', m(x1, x2).shape)          # (1, 2, 256, 256)
print('Train loss:', float(m.compute_loss(x1, x2, torch.zeros(1,256,256).long())))
"
```

### 安装 GPU 版 PyTorch（推荐）

```bash
# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 或 CPU 版本（仅用于代码验证，不推荐训练）
pip install torch torchvision
```

验证安装：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 数据集说明

本项目使用 **WHU-GCD**（武汉大学生成式变化检测数据集）。所有图像统一为 **512×512 PNG** 格式。

### 训练集

| 来源 | 样本数 | 说明 |
|------|--------|------|
| `gcd` | 21,000 | 合成变化样本（来自 3 个语义分割数据集），im2 文件名含类别后缀 `_i` |
| `ugcd_full` | 3,167 | 真实变化 + 负样本（Google Earth），mask 全零 |
| `ucd` | 2,089 | 纯负样本（无变化） |
| `ugcd` | 1,078 | 生成无变化样本 |
| **合计** | **27,334** | |

### 测试集

| 集合 | 样本数 | 支持任务 | 说明 |
|------|--------|---------|------|
| `val` | 600 | BCD | 验证集 |
| `test` | 3,300 | BCD | 合成测试集 |
| `test2` | 3,906 | BCD | 真实世界测试集 |

### 语义类别 ID

| ID | 类别 |
|----|------|
| 2 | 建筑 |
| 3 | 道路 |
| 4 | 水体 |
| 5 | 荒地 |
| 6 | 森林 |
| 7 | 农业 |

### 数据目录结构

```
WHU-GCD/
├── train/
│   ├── gcd/
│   │   ├── im1/        # 前时相图像（14,433 张唯一原图）
│   │   ├── im2/        # 后时相图像（21,000 张，含类别后缀）
│   │   ├── label/      # 二值标签（0=无变化, 255=有变化）
│   │   ├── mask1/      # 前时相语义掩码（值 0–6，WHU-GCD 自带，BCD 不使用）
│   │   └── mask2/      # 后时相语义掩码（值 0–6，WHU-GCD 自带，BCD 不使用）
│   ├── ugcd_full/
│   ├── ucd/
│   └── ugcd/
├── val/
├── test/
└── test2/              # 仅 im1/, im2/, label/
```

> **文件命名规则**：在 `gcd/` 中，`im1/x.png` 对应 `im2/x_i.png`，其中 `i` 为变化类别 ID（2–7）。例如 `E10_0.png`（im1）对应 `E10_0_5.png`（im2），表示类别 5（荒地）发生了变化。

### 数据路径配置（跨服务器迁移）

**本项目所有 partition JSON 已使用相对路径**（如 `train/gcd/im1/x.png`），迁移到新服务器只需指定 `--data_root` 即可，无需修改任何文件。

**迁移步骤：**

1. 将 `FedChange/` 文件夹和 `WHU-GCD/` 数据集上传到新服务器（保持平级即可，或放到任意位置）
2. 在 `FedChange/` 下安装依赖：`pip install -r requirements.txt`
3. 运行实验时通过 `--data_root` 指定数据集位置：

```bash
# 假设新服务器目录结构:
#   /home/user/projects/FedChange/     <-- 代码
#   /home/user/datasets/WHU-GCD/        <-- 数据集

# 联邦实验（Dirichlet Non-IID 划分；集中式训练作上界）
python -m fed_cd.federated.fed_main --data_root /home/user/datasets/WHU-GCD --partition_json partitions/partition_dirichlet_a0.5_n7.json ...
# 集中式上界（全部数据集中训练）
python -m fed_cd.centralized.cen_main --data_root /home/user/datasets/WHU-GCD --project_name Centr_base_bcd ...
```

> **默认值**：`--data_root ../WHU-GCD`（即 FedChange 与 WHU-GCD 平级）。若数据集在其他位置，只需改这一个参数。

> **重新生成划分**：如果在新服务器上需要用不同参数重新生成 partition JSON：
> ```bash
> python -m partitions.generate --data_root /home/user/datasets/WHU-GCD --alpha 0.3 --num_clients 10
> ```
> 新生成的 JSON 同样使用相对路径，跨机器可移植。

---

## Non-IID 数据划分

联邦学习的核心挑战是数据异构性（Non-IID）。本项目使用 **Dirichlet 分布**划分数据：用一个浓度参数 α 精确、连续地控制客户端间的类别分布异构度，从而系统地研究异构度对联邦学习性能的影响。

### 划分方法：Dirichlet 分布（`partitions/generate.py`）

对每个语义类，从 Dirichlet(α, …, α) 采样得到各客户端的分配比例，再按比例把该类样本分给客户端。**类 0（无变化）作为独立的语义类**，与 6 个变化类 {2..7} 一视同仁地参与采样——负样本（ucd / ugcd / ugcd_full）标记为类 0，与 gcd 变化类一起进入 Dirichlet 抽签。

- **α 参数说明**：
  - α → 0：极端异构（每个客户端几乎只持有少数类别）
  - α = 0.5：中度异构
  - α = 1.0：轻度异构
  - α → ∞：接近 IID（均匀划分）
- **适用场景**：扫描多档 α，研究从极端异构到近 IID 的连续性能变化

```bash
python -m partitions.generate --alpha 0.1  --num_clients 7 --data_root ../WHU-GCD   # 极端异构
python -m partitions.generate --alpha 0.5  --num_clients 7 --data_root ../WHU-GCD   # 中度异构
python -m partitions.generate --alpha 1.0  --num_clients 7 --data_root ../WHU-GCD   # 轻度异构
python -m partitions.generate --alpha 100  --num_clients 7 --data_root ../WHU-GCD   # 接近 IID

python -m partitions.generate --alpha 0.5 --no_neg              # 排除负样本（ucd/ugcd/ugcd_full）
python -m partitions.generate --alpha 0.5 --dry_run             # 仅预览统计，不写文件
```

### 已生成的划分文件

| 文件 | 客户端数 | 总样本 | α | 异构程度 |
|------|---------|--------|---|---------|
| `partition_dirichlet_a0.1_n7.json` | 7 | 27,334 | 0.1 | 极端统计异构 |
| `partition_dirichlet_a0.5_n7.json` | 7 | 27,334 | 0.5 | 中度统计异构 |
| `partition_dirichlet_a1.0_n7.json` | 7 | 27,334 | 1.0 | 轻度统计异构 |
| `partition_dirichlet_a100.0_n7.json` | 7 | 27,334 | 100.0 | 近 IID（基线） |

> 这些 JSON 文件被 gitignore（每个约 8 MB，可按需用上面的命令重生）。命名规则：`partition_dirichlet_a{α}_n{客户端数}.json`。

### 重新生成划分

```bash
# 查看划分统计（不写文件）
python -m partitions.generate --alpha 0.5 --dry_run

# 生成新参数的划分
python -m partitions.generate --alpha 0.3 --num_clients 10
```

---

## 快速开始

以下命令均在 `FedChange/` 目录下执行。

### 1. 验证安装

```bash
python -c "
from fed_cd.models.bit_cd import build_bit_cd_model
import torch
model = build_bit_cd_model(num_classes=2, pretrained=False)
x1, x2 = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
print('Output:', model(x1, x2).shape)
print('Params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
"
```

所有运行脚本（`run_all.sh` 及各 `run_*.sh`）启动时会**自动检查依赖**：缺核心库或缺 torchange 基线库（仅 torchange_fed / changen2 步骤需要）时，会打印中文提示与安装命令并中止，不会跑到一半才在深处报 `ImportError`。也可手动调用检查：

```bash
source scripts/check_env.sh
check_env_core          # 核心库（所有实验都需要）
check_env_torchange     # torchange 基线库（可选，仅 torchange_fed / changen2 需要）
```

### 2. 快速运行联邦实验（2 轮验证）

```bash
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a0.5_n7.json \
    --data_root ../WHU-GCD \
    --epochs 2 \
    --frac_num 2 \
    --local_ep 1 \
    --local_bs 4 \
    --img_size 128 \
    --pretrained False \
    --eval_splits val \
    --global_test_frequency 2 \
    --project_name quick_test
```

---

## 集中式训练（性能上界）

集中式训练（centralized）将全部训练样本（gcd + ugcd_full + ucd + ugd，共 27,334 张）喂给单个 BIT-CD 模型训练，作为联邦实验的**性能上界**。它与联邦实验**共享同一训练集、同一模型、同一超参和同一评估流程**，唯一区别是集中式 vs. 联邦优化，因此可直接对比，闭环为：

```
Centralized（上界） ≥ Fed-IID ≥ Fed-NonIID
```

### 运行集中式实验

```bash
# 默认超参与联邦 BIT-CD 实验对齐（net_G / epochs / batch_size / lr 等）
bash scripts/run_centralized.sh
# 或
python -m fed_cd.centralized.cen_main \
    --data_root ../WHU-GCD \
    --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 --img_size 256 \
    --epochs 200 --batch_size 8 --lr 0.01 \
    --lr_policy linear --optimizer sgd --pretrained True \
    --eval_splits val,test,test2 \
    --project_name Centr_base_bcd \
    --checkpoint_root results/centralized \
    --seed 42
```

输出 `results/centralized/<project_name>/results.json`，schema 与联邦实验完全一致，`summarize.py` 零改动兼容。也支持 torchange 基线（如 `bash scripts/run_centralized.sh 200 8 256 0.01 changesparse_bcd changesparse`）。

---

## 联邦学习训练

> **说明**：性能上界由**集中式训练**（`fed_cd.centralized.cen_main`）提供；近 IID 的联邦训练（`partition_dirichlet_a100.0_n7.json`，α=100 近均匀）作为近上界参考，与各 Non-IID 档同框架对比。

### 联邦算法

本项目支持 4 种联邦优化算法，统一通过 `--fed_alg` 参数路由：

| 算法 | `--fed_alg` | 关键参数 | 原理 |
|------|------------|---------|------|
| **FedAvg** | `fedavg` | `--iid True/False` | 经典联邦平均；`--iid True` 简单平均，`False` 按数据量加权（默认） |
| **FedProx** | `fedprox` | `--fedprox_mu 0.01` | 本地损失加近端正则项，约束客户端不偏离全局模型 |
| **FedNova** | `fednova` | — | 按归一化本地步数加权聚合，解决异构本地 epoch 数的优化不一致 |
| **SCAFFOLD** | `scaffold` | `--scaffold_lr 1.0` | 控制变量法修正客户端漂移（client drift），收敛更快 |
| **全局 EMA** | （叠加项） | `--globalema True` | 对全局参数做指数移动平均，可与任意算法叠加提升稳定性 |

> **算法兼容性**：FedAvg / FedProx / FedNova / EMA 对所有模型（BIT-CD + torchange 基线）兼容。**SCAFFOLD 仅支持 BIT-CD**——torchange 基线的损失是内置黑盒（BCE+Dice），无法可靠计算 control variate，检测到 torchange 模型时会打印警告并自动降级为 FedAvg。

> **向后兼容**：旧脚本只传 `--fedprox_mu>0`（未指定 `--fed_alg`）时仍会触发 proximal 项，行为与旧版一致。

### 运行单个联邦实验

```bash
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_dirichlet_a0.5_n7.json \
    --data_root ../WHU-GCD \
    --net_G base_transformer_pos_s4_dd8 \
    --num_classes 2 \
    --img_size 256 \
    --epochs 200 \
    --frac_num 5 \
    --local_ep 2 \
    --local_bs 8 \
    --lr 0.01 \
    --lr_policy linear \
    --fed_alg fedprox --fedprox_mu 0.01 \
    --iid False \
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --project_name "FedProx_dirichlet05_bcd_s42"
```

> `project_name` 内嵌 `_s<seed>` 后缀（见 [多 seed 实验](#多随机种子实验)），避免不同 seed 的 `results.json` 互相覆盖。

### 批量运行 BCD 实验（Dirichlet α 扫描）

一次性运行 FedAvg 和 FedProx 在 4 档 Dirichlet α（0.1/0.5/1.0/100）上的全部实验（默认 3 个 seed：42 / 2024 / 0）：

```bash
# 位置参数末尾可自定义 seed 列表
bash scripts/run_fed_bcd.sh 200 5 2 8 256 0.01 0.01 "42 2024 0"
```

> 位置参数顺序：`epochs frac_num local_ep local_bs img_size lr fedprox_mu seeds_str`

### 多随机种子实验

为支持统计显著性（FL 实验对客户端采样随机性敏感），所有批量脚本均内置多 seed 循环：

- **默认 3 seed**：`42 2024 0`
- **产物隔离**：每个实验的 `project_name` 内嵌 `_s<seed>` 后缀（如 `FedAvg_dirichlet_05_bcd_s42`），`results.json` 落在独立子目录，互不覆盖
- **自动聚合**：`summarize.py` 自动按 `_s<seed>` 剥离后缀分组，输出 mean±std 表（见 [评估与结果汇总](#评估与结果汇总)）

```bash
# 单脚本多 seed
bash scripts/run_fed_alg_comparison.sh partitions/partition_dirichlet_a0.5_n7.json "42 2024 0"
```

### 算法对比实验（4 算法）

在固定 Non-IID 划分（默认 Dirichlet α=0.5，异构度适中、算法差异最显著）上对比 4 种联邦算法：

```bash
# 4 算法 × 3 seed = 12 runs
bash scripts/run_fed_alg_comparison.sh
```

输出到 `results/alg_comparison/`，汇总：`python -m fed_cd.summarize --results_root results/alg_comparison`。

### 一键全量实验（总入口）

`scripts/run_all.sh` 是**总入口**：传一个 seed，依次调用全部 5 类实验子脚本，每个都用这同一个 seed。这是最常用的"单 seed 全量实验"工作流。

```bash
# 用某个 seed 跑完全部实验
bash scripts/run_all.sh 42

# 跑 3 个 seed 的完整工作流（推荐）
for s in 42 2024 0; do bash scripts/run_all.sh $s; done
python -m fed_cd.summarize          # 汇总全部，自动产出 mean±std 表
```

选择性执行（步骤开关）与公共超参覆盖：

```bash
bash scripts/run_all.sh 42 --only alg_comparison              # 只跑单个对比试验
bash scripts/run_all.sh 42 --only fed_bcd,torchange_fed       # 跑指定的若干个
bash scripts/run_all.sh 42 --skip centralized,changen2        # 跳过这两步
bash scripts/run_all.sh 42 --epochs 100 --bs 16 --lr 0.005    # 覆盖公共超参
```

执行的 5 类实验（按顺序，可用 `--only/--skip` 的步骤名见左列）：

| 步骤名 | 脚本 | 内容 | 输出目录 |
|--------|------|------|---------|
| `centralized` | `run_centralized.sh` | 集中式上界 | `results/centralized/` |
| `alg_comparison` | `run_fed_alg_comparison.sh` | 4 联邦算法对比 | `results/alg_comparison/` |
| `fed_bcd` | `run_fed_bcd.sh` | 4 个 Dirichlet α × FedAvg+FedProx 主矩阵 | `results/fed_bcd/` |
| `torchange_fed` | `run_fed_torchange_bcd.sh` | BIT-CD vs torchange 基线 | `results/torchange_fed/` |
| `changen2` | `run_changen2_zeroshot.py` | 零样本评估（免训练） | `results/torchange/` |

> **健壮性**：单步实验失败会打印 `[FAIL]` 提示并继续后续步骤（适合长时间批跑，避免一个实验挂掉让后面全白费）。

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fed_alg` | fedavg | 联邦算法：`fedavg` / `fedprox` / `fednova` / `scaffold` |
| `--epochs` | 200 | 全局通信轮数 |
| `--frac_num` | 5 | 每轮参与训练的客户端数量 |
| `--local_ep` | 2 | 每个客户端本地训练的 epoch 数 |
| `--local_bs` | 8 | 本地训练 batch size |
| `--img_size` | 256 | 输入图像尺寸（从 512 缩放） |
| `--lr` | 0.01 | 初始学习率 |
| `--lr_policy` | linear | 学习率衰减策略 |
| `--fedprox_mu` | 0.0 | FedProx 近端项权重（仅 `--fed_alg fedprox` 时生效） |
| `--scaffold_lr` | 1.0 | SCAFFOLD 服务端 control variate 更新步长（仅 `--fed_alg scaffold`） |
| `--pretrained` | True | 是否使用 ImageNet 预训练 ResNet 权重 |
| `--iid` | False | FedAvg 聚合方式：True=简单平均, False=加权平均 |
| `--globalema` | False | 是否叠加全局 EMA（可与任意算法组合） |
| `--seed` | 42 | 随机种子（建议跑 3 个：42 / 2024 / 0） |

---

## torchange 对比基线

为系统对比 FedChange 框架（BIT-CD）的性能，本项目集成了 [torchange](https://github.com/Z-Zheng/pytorch-change-models) 库中的多个 SOTA 变化检测模型作为基线。这些基线经适配器在**联邦**框架下（Non-IID 鲁棒性，框架核心贡献）与 BIT-CD 直接对比。

### 可用基线

| `--net_G` 名称 | 模型 | 论文 | 骨干 | 参数量 | 任务 |
|----------------|------|------|------|--------|------|
| `changesparse_bcd` | ChangeSparseBCD | ISPRS 2024 (PCM) | er.R18 | ~13.0M | 二值 CD |
| `changestar_1xd_r18` | ChangeStar1xd | ICCV 2021 | FarSeg ResNet18 | ~16.4M | 二值 CD（R18 受控消融） |
| `changestar_1xd` | ChangeStar1xd | ICCV 2021 | FarSeg ResNet50 | ~31.0M | 二值 CD |
| `changestar_2_5` | ChangeStar2.5 | IJCV 2024 | FarSeg ResNet50 | ~34.1M | 二值 CD（现代版） |
| `changen2_zeroshot` | Changen2 ChangeStar1x256 | TPAMI 2024 | ViT-B | — | **零样本**（免训练参考点） |

> 注：参数量为 `pretrained=False` 下的实测值；`changen2_zeroshot` 会从 HuggingFace 自动下载 Changen2-S1 预训练权重。

### 不包含的模型及原因

- **ChangeStar2**：其原生训练流水线是 **G-STAR 单时相监督协议**（需语义掩码 `XMASK1` 并从 T1 合成伪 T2），与本项目 WHU-GCD 的**双时相监督**设定根本不兼容，故排除。
- **ChangeOS / ChangeMask / ChangeSparseO2M/M2M**：面向建筑损毁分级（5 类）或语义变化检测（6 类），非二值 CD。
- **NeDS / RSDiT / AnyChange**：生成式扩散或零样本实例分割，非逐像素二值判别模型。

### 适配器机制

torchange 模型遵循 `ever` 范式（训练时 `forward(x, y)` 返回**损失字典**，BCE+Dice 内置；评估时 `forward(x)` 返回概率），与 FedChange 的 `forward(x1,x2)->logits` + 外部 CrossEntropyLoss 契约不同。`fed_cd/models/torchange_adapter.py` 中的 `TorchangeCDAdapter` 通过两层接口统一：

| 方法 | 用途 | 实现 |
|------|------|------|
| `forward(x1, x2)` | 推理/评估 | 内部 `eval()` 调 torchange，取变化概率 `p`，转为 2 类 logits `[0, logit(p)]` |
| `compute_loss(x1, x2, label)` | 训练 | 内部 `train()` 调 torchange 原生损失，返回标量（BCE+Dice 之和） |

**CE 等价性**：`softmax([0, logit(p)])[1] = sigmoid(logit(p)) = p`，因此 FedChange 的评估器（`argmax(dim=1)`）、混淆矩阵指标**零改动**复用（已验证 `CE == BCE` 精确相等）。训练循环通过 `hasattr(model, 'compute_loss')` 自动分支：BIT-CD 走 CrossEntropyLoss，torchange 走原生损失。

> **公平性说明**：由于 torchange 训练时损失内置不可分离，无法对所有模型统一 CE。各模型损失类型会在汇总表中标注（BIT-CD = CE，torchange = BCE+Dice 原生）。这反而让每个基线以其论文原貌参与对比，是更忠实的基线协议。

### 运行对比实验

#### 1. 联邦对比（Non-IID 鲁棒性，核心贡献）

联邦聚合是**模型无关**的（纯 `state_dict` 平均），torchange 模型经适配器可无缝接入 `fed_main`，在相同 Non-IID 划分与调度下对比：

```bash
# FedAvg + FedProx，Dirichlet α=0.5
bash scripts/run_fed_torchange_bcd.sh
```

脚本默认在 `partition_dirichlet_a0.5_n7.json` 上运行；如需遍历多种异构度，取消脚本中 `PARTITIONS` 数组的注释行即可（可加上 `partition_dirichlet_a0.1_n7.json` / `partition_dirichlet_a100.0_n7.json`）。真正的性能上界用 `bash scripts/run_centralized.sh` 跑集中式版本。

#### 2. Changen2 零样本评估（免训练参考点）

Changen2 是变化检测基础模型，在 Changen2-S1-15k 上预训练。直接在 WHU-GCD 上**零样本**评估（不微调）：

```bash
python scripts/run_changen2_zeroshot.py --data_root ../WHU-GCD
```

> 该脚本对输入做了 **ImageNet 归一化转换**（CDDataset 默认 0.5/0.5，而 Changen2 ViT 骨干需 ImageNet 统计量），是精确线性变换。

### 骨干受控消融

`changesparse_bcd`（er.R18）与 `changestar_1xd_r18`（FarSeg R18）均采用 ResNet18，与 BIT-CD（ResNet18）骨干一致，可隔离"变化建模"与"骨干"的贡献，构成受控对比。

---



### 评估指标

所有指标基于像素级混淆矩阵计算：

| 指标 | 说明 |
|------|------|
| **mF1** (mean F1) | 各类 F1 的平均值，**主要指标** |
| **mIoU** (mean IoU) | 各类交并比的平均值 |
| **F1_change** | 变化类的 F1（类别 1） |
| **IoU_change** | 变化类的 IoU（类别 1） |
| **Precision / Recall** | 各类的精确率和召回率 |
| **OA** (Overall Accuracy) | 总体像素精度 |

### 评估协议

训练过程中，模型会在以下三个测试集上评估：

1. **val**（验证集）— 用于选择最佳模型
2. **test**（合成测试集）— 评估合成数据上的性能
3. **test2**（真实测试集）— 评估真实数据上的泛化性能

### 结果汇总

训练完成后，所有结果保存在 `results/<project_name>/results.json` 中。使用汇总工具生成对比表：

```bash
python -m fed_cd.summarize
```

输出示例（含跨模型对比表，标注 `net_G` 与损失类型以保证对比公平性）：

```
[ Cross-model comparison (final results, best ckpt by val mF1) ]
Experiment                              net_G                  loss               |          val           |          test          |         test2
                                                                                   |     mF1    mIoU    F1_c |     mF1    mIoU    F1_c |     mF1    mIoU    F1_c
=========================================================================================================================================================
Centralized_A                           base_transformer...    CE                 |  89.32%  82.15%  85.67% |  87.45%  80.12%  83.23% |  76.89%  68.34%  71.56%
Centralized_changesparse_bcd            changesparse_bcd       BCE+Dice(native)   |  88.10%  80.90%  84.20% |  86.30%  79.00%  82.10% |  75.40%  66.80%  70.10%
Centralized_changestar_2_5              changestar_2_5         BCE+Dice(native)   |  90.20%  83.40%  86.80% |  88.50%  81.20%  84.30% |  77.90%  69.40%  72.70%
```

`summarize` 还会自动识别多 seed 实验（`_s<seed>` 后缀），输出 mean±std 表：

```
[ Multi-seed mean ± std (final results) ]
Experiment (seeds)               |          test (mF1±std)         |         test2 (mF1±std)
FedAvg_dirichlet_05_bcd (n=3)    |       82.33±1.04%               |       74.00±1.20%
FedProx0.01_dirichlet_05_bcd (n=3)|      84.30±0.62%               |       76.20±0.75%
```

> ⚠️ **口径说明**：`summarize` 输出的"per-metric best"表是**每指标独立取历史最优的 oracle 上界**（会高估，等同在测试集上挑最好那一轮），仅供收敛分析，**不可用于论文主表**。论文报告只能用 `final_results`（基于 val mF1 选模的单次结果）或上表的 mean±std。该表会打印 `ORACLE upper bound — NOT for paper reporting` 警示。

### 可视化

`fed_cd/plot.py` 一键生成论文级图表，数据来自各 `results.json` 的 `eval_history` / `train_loss_history`：

```bash
# 1. 收敛曲线（metric vs 轮次，多算法叠加；多 seed 聚合为 mean±std 阴影带）
python -m fed_cd.plot --results_root results/alg_comparison \
    --convergence --metric mf1 --split test --out results/figs

# 2. 训练 loss 曲线
python -m fed_cd.plot --train_loss --results_root results/alg_comparison

# 3. Dirichlet α 扫描（log-x 横轴，带 errorbar，按算法分组）
python -m fed_cd.plot --alpha_sweep --results_root results/fed_bcd --split test

# 4. 预测结果可视化（TP=绿 / FP=红 / FN=蓝 / TN=黑 着色，T1+T2+预测三行叠加）
python -m fed_cd.plot --predictions \
    --ckpt results/alg_comparison/.../best_ckpt.pt \
    --net_G base_transformer_pos_s4_dd8 \
    --data_root ../WHU-GCD --split test --n 8
```

| 图表 | 数据来源 | 用途 |
|------|---------|------|
| 收敛曲线 | `eval_history` | 对比算法收敛速度与最终性能（FL 论文必备） |
| 训练 loss 曲线 | `train_loss_history` | 辅助收敛性分析，常与收敛曲线并排 |
| Dirichlet α 扫描 | `final_results` + 实验名 | 展示异构度对性能的影响趋势 + seed 稳健性 |
| 预测着色图 | checkpoint 推理 | 定性展示检测质量（遥感 CD 论文标配） |

> 多 seed 聚合：收敛曲线和 α 扫描会自动按 `_s<seed>` 分组，画均值曲线 + ±1σ 阴影/误差棒。需要 `matplotlib`（见 [环境配置](#环境配置)）。

### 单独评估已训练模型

```bash
python -c "
import torch
from fed_cd.models.bit_cd import build_bit_cd_model
from fed_cd.evaluation.evaluator import evaluate_model
from fed_cd.data.cd_dataset import CDDataset
from fed_cd.data.data_partition import scan_evaluation_set
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_bit_cd_model(num_classes=2, pretrained=False).to(device)
ckpt = torch.load('results/FedAvg_source_bcd/best_ckpt.pt', map_location=device)
model.load_state_dict(ckpt['model'])

test_samples = scan_evaluation_set('../WHU-GCD', 'test')
ds = CDDataset(test_samples, img_size=256, is_train=False)
loader = DataLoader(ds, batch_size=1, shuffle=False)

scores = evaluate_model(model, loader, device, n_class=2)
print(f'Test mF1={scores[\"mf1\"]:.4f}, mIoU={scores[\"miou\"]:.4f}')
"
```

---

## 技术架构

### BIT-CD 模型

BIT-CD（Bitemporal Image Transformer）的变化检测流程：

```
前时相图像 x1 ─┐                ┌─ 后时相图像 x2
               ↓                ↓
        ResNet18 Backbone (共享权重, 1/8 降采样)
               │                │
         特征图 f1              特征图 f2
               │                │
      语义 Tokenizer (空间注意力)
               │                │
         Token t1              Token t2
               └──── 拼接 ──────┘
                       │
              Transformer Encoder (自注意力)
                       │
               t1', t2' (分裂)
                       │                │
         Transformer Decoder       Transformer Decoder
               │                        │
         增强特征 f1'             增强特征 f2'
               └──── 差异 ──────┘
                       │
                 |f1' - f2'|
                       │
              上采样 → 分类器
                       │
                输出: 变化概率图
```

**模型参数**：12.4M（ResNet18 backbone + Transformer）

**模型变体**：

| 名称 | 说明 |
|------|------|
| `base_resnet18` | 仅 ResNet18，无 Transformer |
| `base_transformer_pos_s4` | BIT + 1 层解码器 |
| `base_transformer_pos_s4_dd8` | BIT + 8 层解码器（默认，性能最佳） |
| `base_transformer_pos_s4_dd8_dedim8` | BIT + 8 层解码器，注意力维度=8 |

> 除上述 BIT-CD 变体外，`--net_G` 还支持 torchange 基线（`changesparse_bcd`、`changestar_1xd`、`changestar_1xd_r18`、`changestar_2_5`、`changen2_zeroshot`）。所有模型经统一工厂 `fed_cd.models.build_cd_model()` 构建，详见 [torchange 对比基线](#torchange-对比基线)。

### 联邦学习流程

```
┌──────────────────────────────────────────────────┐
│                   服务器端 (Server)                │
│                                                    │
│  for round t = 1, 2, ..., T:                      │
│    1. 选择 frac_num 个客户端                       │
│    2. 广播全局模型 w_t 给选中客户端                 │
│    3. 等待客户端完成本地训练                        │
│    4. 聚合权重:                                    │
│       - FedAvg:  w_{t+1} = Σ w_k / K              │
│       - 加权:    w_{t+1} = Σ (n_k * w_k) / N     │
│    5. （可选）EMA 平滑                              │
│    6. 每 freq 轮在 val/test 上评估                  │
│                                                    │
└──────────────────────────────────────────────────┘
           ↑ w_t                    ↓ w_t
    ┌──────┴──────┐          ┌──────┴──────┐
    │  客户端 k    │          │  客户端 k    │
    │              │          │              │
    │  本地数据 D_k│          │  本地数据 D_k│
    │              │          │              │
    │  for e = 1..E:          │  (FedProx)   │
    │    loss = CE(f(x), y)   │  + μ/2·||w-w_t||²│
    │    w_k ← w_k - lr·∇loss │              │
    │              │          │              │
    │  返回 w_k    │          │  返回 w_k    │
    └──────────────┘          └──────────────┘
```

### 数据增强

双时相数据增强必须同步（相同的翻转/裁剪/模糊参数应用于 im1 和 im2），以保持空间对应关系：

- 水平随机翻转
- 垂直随机翻转
- 缩放随机裁剪（scale 1.0–1.2 → crop 到目标尺寸）
- 随机高斯模糊
- 归一化（mean=0.5, std=0.5）

---

## 实验设计

### 推荐实验矩阵

#### Phase 1：基线验证

| 实验 | 划分 | 算法 | 目的 |
|------|------|------|------|
| **集中式（上界）** | 全部数据集中训练 | `cen_main`（非联邦） | 性能上界 |
| 联邦近 IID | Dirichlet α=100 | FedAvg | 近 IID 参考 |

#### Phase 2：Non-IID 对比（FedAvg vs FedProx）

> 每格 ×3 seed（42/2024/0），汇总 mean±std。运行：`bash scripts/run_fed_bcd.sh`。

| 划分 | FedAvg | FedProx (μ=0.01) |
|------|--------|------------------|
| Dirichlet α=0.1 | ✓ | ✓ |
| Dirichlet α=0.5 | ✓ | ✓ |
| Dirichlet α=1.0 | ✓ | ✓ |
| Dirichlet α=100 | ✓ | ✓ |

#### Phase 2b：联邦算法对比（SOTA）

在固定 Non-IID 划分（Dirichlet α=0.5）上对比 4 种联邦算法，验证本框架对各 SOTA 算法的支持。每格 ×3 seed。

| 算法 | `--fed_alg` | BIT-CD | torchange 基线 |
|------|------------|--------|---------------|
| FedAvg | `fedavg` | ✓ | ✓ |
| FedProx | `fedprox` | ✓ | ✓ |
| FedNova | `fednova` | ✓ | ✓ |
| SCAFFOLD | `scaffold` | ✓ | 自动降级为 FedAvg（黑盒损失） |

> 运行：`bash scripts/run_fed_alg_comparison.sh`。详见 [算法对比实验](#算法对比实验4-算法)。

#### Phase 3：消融实验

| 消融维度 | 变化范围 | 研究问题 |
|---------|---------|---------|
| Dirichlet α | 0.1, 0.5, 1.0, 100 | 数据异构度的影响 |
| FedProx μ | 0, 0.001, 0.01, 0.1, 1.0 | 近端约束强度 |
| local_ep | 1, 2, 5 | 本地训练强度 |
| frac_num | 1, 5, 7（全选） | 客户端参与率（直接改 `--frac_num`） |
| img_size | 128, 256, 512 | 输入分辨率 |

#### Phase 4：跨模型对比（BIT-CD vs torchange 基线）

在相同 Non-IID 划分与调度下，对比 BIT-CD 与 torchange 基线的联邦鲁棒性：

| 模型 | 联邦（α=0.5, FedAvg+FedProx） | 骨干 |
|------|-------------------------------|------|
| BIT-CD（本框架） | ✓ | R18 |
| ChangeSparseBCD | ✓ | R18（受控） |
| ChangeStar1xd (R18) | ✓ | R18（受控消融） |
| ChangeStar1xd (R50) | ✓ | R50 |
| ChangeStar2.5 | ✓ | R50 |
| Changen2 零样本 | —（仅零样本评估） | ViT-B |

> 运行脚本：`scripts/run_fed_torchange_bcd.sh`、`scripts/run_changen2_zeroshot.py`。详见 [torchange 对比基线](#torchange-对比基线)。

### 结果分析维度

| 分析 | 关键指标 |
|------|---------|
| 异构度影响 | Dirichlet α 从 100→0.1 的性能变化 |
| 算法对比 | FedAvg vs FedProx 在不同划分下的差异 |
| 合成 vs 真实 | test vs test2 性能差距 |
| 通信效率 | 达到 90% 近 IID 性能所需通信轮数 |
| 负样本影响 | 负样本客户端对全局模型的贡献 |

---

## 常见问题

### Q: CUDA 不可用怎么办？

确认安装了 GPU 版本的 PyTorch。CPU 版本可用于代码验证，但训练速度约为 GPU 的 1/50。

### Q: 内存不足（OOM）？

- 减小 `--local_bs`（如 4 或 2）
- 减小 `--img_size`（如 128）
- 使用 `--pretrained False` 跳过下载预训练权重

### Q: 如何添加新的划分参数？

当前划分统一由 `partitions/generate.py` 的 Dirichlet 方法生成，通过 `--alpha` / `--num_clients` 控制异构度与客户端数。如需新的划分方法：

1. 在 `partitions/` 包内新增模块（如 `partitions/my_method.py`）
2. 复用 `from partitions.generate import scan_all_sources, build_partition_output, save_partition` 等公共工具
3. 输出标准 JSON 格式（参考已有划分文件结构）
4. 每个样本包含 `im1, im2, label, source` 字段（`mask1, mask2` 可选，BCD 训练不读取）

### Q: 如何适配其他变化检测模型？

本项目通过统一工厂 `fed_cd.models.build_cd_model()` 调度模型。新增模型有两种方式：

1. **BIT-CD 风格**（外部损失）：模型满足 `forward(x1, x2)` 返回 logits `(B, num_classes, H, W)`，由训练循环施加 CrossEntropyLoss。在 `build_cd_model()` 中注册即可。

2. **torchange / ever 风格**（内置损失）：参考 `fed_cd/models/torchange_adapter.py` 中的 `TorchangeCDAdapter`，实现 `forward(x1, x2)`（推理返回 2 类 logits）与 `compute_loss(x1, x2, label)`（训练返回标量损失），训练循环会通过 `hasattr(model, 'compute_loss')` 自动走原生损失分支。联邦聚合基于 `state_dict`，对模型内部结构无要求。

---

## 引用

如果你使用了本项目，请引用以下论文：

```bibtex
@article{chen2021remote,
  title={Remote Sensing Image Change Detection with Transformers},
  author={Chen, Hao and Qi, Zipeng and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={59},
  number={12},
  pages={10213--10226},
  year={2021}
}

@inproceedings{miao2023fedseg,
  title={FedSeg: Class-Heterogeneous Federated Learning for Semantic Segmentation},
  author={Miao, Jiaxu and Yang, Zongxin and Fan, Leilei and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8042--8052},
  year={2023}
}
```

如果使用了 torchange 对比基线，请额外引用：

```bibtex
@article{zheng2024unifying,
  title={Unifying Remote Sensing Change Detection via Deep Probabilistic Change Models: from Principles, Models to Applications},
  author={Zheng, Zhuo and others},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2024}
}

@article{zheng2024single,
  title={Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection},
  author={Zheng, Zhuo and others},
  journal={International Journal of Computer Vision},
  year={2024}
}

@article{zheng2024changen2,
  title={Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model},
  author={Zheng, Zhuo and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

WHU-GCD 数据集相关问题请联系：yujiezan@whu.edu.cn

---

## License

本项目代码仅供非商业和研究目的使用。
