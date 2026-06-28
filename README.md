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

- **模型无关的联邦框架**：权重聚合（FedAvg / 加权 FedAvg / FedProx）与模型结构完全解耦，可适配任意 PyTorch 模型
- **4 种 Non-IID 数据划分策略**：按来源（域偏移）、按语义类别、Dirichlet 分布、混合划分，全面覆盖遥感场景中的异构性
- **双任务支持**：二值变化检测（BCD，2 类）和语义变化检测（SCD，6 类地物变化）
- **BIT-CD Transformer 模型**：自包含实现，基于 ResNet18 + Bitemporal Transformer，12.4M 参数
- **torchange 对比基线**：集成 [torchange](https://github.com/Z-Zheng/pytorch-change-models) 库的 4 个 SOTA 变化检测模型 + 1 个零样本基础模型，支持联邦对比（详见 [torchange 对比基线](#torchange-对比基线)）
- **完整的评估体系**：F1、IoU、Precision、Recall，在合成测试集和真实测试集上分别评估
- **集中式训练上界**：提供集中式 BIT-CD 训练（`fed_cd.centralized.cen_main`）作为性能上界，与联邦实验同框架、同超参对比
- **IID 基线**：提供分层随机 IID 划分作为联邦实验的近上界参考

---

## 项目结构

```
FedChange/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
│
├── partition_utils.py                 # 数据划分公共工具（扫描、解析、统计）
├── partition_by_source.py             # 策略 1：按数据来源划分（域偏移）
├── partition_dirichlet.py             # 策略 2：Dirichlet 分布划分（可控异构度）
├── partition_hybrid.py                # 策略 3：混合划分（域偏移 + 类异构）
├── partition_noniid.py                # 策略 4：Non-IID1/2 类别异构（固定 K，对比实验用）
├── partition_iid.py                   # 分层随机 IID（Non-IID1/2 的对比基线）
│
├── partitions/                        # 已生成的 Non-IID 划分文件（JSON）
│   ├── partition_source.json
│   ├── partition_dirichlet_a0.1_n7.json
│   ├── partition_dirichlet_a0.5_n7.json
│   ├── partition_dirichlet_a1.0_n7.json
│   ├── partition_dirichlet_a100.0_n7.json
│   ├── partition_hybrid_c1_separate.json
│   ├── partition_noniid1_K70.json
│   ├── partition_noniid2_K70.json
│   └── partition_iid_K70.json
│
├── fed_cd/                            # 核心代码包
│   ├── options.py                     # 统一命令行参数管理
│   ├── summarize.py                   # 实验结果汇总工具
│   │
│   ├── data/                          # 数据处理
│   │   ├── cd_dataset.py              #   WHU-GCD 双时相数据集（BCD/SCD）
│   │   ├── data_partition.py          #   划分加载 + 评估集扫描
│   │   └── data_utils.py              #   同步双时相数据增强
│   │
│   ├── models/                        # 模型
│   │   ├── bit_cd.py                  #   BIT-CD 模型（ResNet18 + Transformer）
│   │   ├── torchange_adapter.py       #   torchange 基线适配器（统一 forward/compute_loss 接口）
│   │   └── __init__.py                #   统一模型工厂 build_cd_model()
│   │
│   ├── federated/                     # 联邦学习
│   │   ├── fed_main.py                #   联邦主循环
│   │   ├── local_update.py            #   本地训练（含 FedProx）
│   │   └── aggregation.py             #   权重聚合 + EMA
│   ├── centralized/                   # 集中式训练（性能上界）
│   │   └── cen_main.py                #   集中式主循环（与联邦同框架对比）
│   │
│   └── evaluation/                    # 评估
│       ├── cd_metrics.py              #   混淆矩阵指标计算
│       └── evaluator.py               #   多集评估器
│
├── scripts/                           # 运行脚本
│   ├── run_fed_bcd.ps1                # BCD 联邦实验（PowerShell）
│   ├── run_fed_bcd.sh                 # BCD 联邦实验（Bash）
│   ├── run_fed_torchange_bcd.ps1      # torchange 联邦对比（PowerShell）
│   ├── run_fed_torchange_bcd.sh       # torchange 联邦对比（Bash）
│   ├── run_fed_class_comparison.ps1   # Non-IID1/2/IID 类别对比（PowerShell）
│   ├── run_fed_class_comparison.sh    # Non-IID1/2/IID 类别对比（Bash）
│   ├── run_centralized.ps1            # 集中式训练上界（PowerShell）
│   ├── run_centralized.sh             # 集中式训练上界（Bash）
│   ├── run_fed_fracnum_sweep.ps1      # frac_num 参与率稳健性 sweep（PowerShell）
│   ├── run_fed_fracnum_sweep.sh       # frac_num 参与率稳健性 sweep（Bash）
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
| `val` | 600 | BCD + SCD | 验证集 |
| `test` | 3,300 | BCD + SCD | 合成测试集 |
| `test2` | 3,906 | 仅 BCD | 真实世界测试集（无语义掩码） |

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
│   │   ├── mask1/      # 前时相语义掩码（值 0–6）
│   │   └── mask2/      # 后时相语义掩码（值 0–6）
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

# 联邦实验（集中式训练作上界，IID 联邦作近上界对比）
python -m fed_cd.federated.fed_main --data_root /home/user/datasets/WHU-GCD --partition_json partitions/partition_iid_K70.json ...
# 集中式上界（全部数据集中训练）
python -m fed_cd.centralized.cen_main --data_root /home/user/datasets/WHU-GCD --project_name Centr_base_bcd ...
```

> **默认值**：`--data_root ../WHU-GCD`（即 FedChange 与 WHU-GCD 平级）。若数据集在其他位置，只需改这一个参数。

> **重新生成划分**：如果在新服务器上需要用不同参数重新生成 partition JSON：
> ```bash
> python partition_dirichlet.py --data_root /home/user/datasets/WHU-GCD --alpha 0.3 --num_clients 10
> ```
> 新生成的 JSON 同样使用相对路径，跨机器可移植。

> **路径迁移工具**：如果 JSON 中包含旧的绝对路径（从其他项目复制的），用以下命令一键转换为相对路径：
> ```bash
> python migrate_paths.py          # 转换所有 JSON
> python migrate_paths.py --check  # 仅检查，不修改
> ```

---

## Non-IID 数据划分

联邦学习的核心挑战是数据异构性（Non-IID）。本项目实现了 4 种划分策略，模拟遥感场景中不同的异构来源。

### 划分策略详解

#### 策略 1：按数据来源划分（`partition_by_source.py`）

将 4 种数据来源（gcd / ugcd_full / ucd / ugcd）分配给不同客户端，利用合成数据 vs 真实数据的天然域差异模拟 Non-IID。

```bash
python partition_by_source.py --clients_per_source 3,2,1,1
```

- **模拟场景**：不同机构各持有不同来源的数据（如有的机构只有合成数据，有的只有真实数据）
- **Non-IID 类型**：域偏移（Domain Shift）
- **默认划分**：7 个客户端（gcd×3, ugcd_full×2, ucd×1, ugcd×1）

#### 策略 2：Dirichlet 分布划分（`partition_dirichlet.py`）

使用 Dirichlet 分布精确控制客户端之间的类别分布异构程度。

```bash
python partition_dirichlet.py --alpha 0.1 --num_clients 7    # 极端异构
python partition_dirichlet.py --alpha 0.5 --num_clients 7    # 中度异构
python partition_dirichlet.py --alpha 1.0 --num_clients 7    # 轻度异构
python partition_dirichlet.py --alpha 100 --num_clients 7    # 接近 IID
```

- **α 参数说明**：
  - α → 0：极端异构（每个客户端几乎只有 1 个类别）
  - α = 0.5：中度异构
  - α → ∞：接近 IID（均匀划分）
- **适用场景**：系统性地研究异构度对联邦学习性能的影响

#### 策略 3：混合划分（`partition_hybrid.py`）

同时模拟域偏移和类别异构两种 Non-IID 场景。

```bash
python partition_hybrid.py --domain_clients 3,2 --classes_per_client 1
```

- 先将数据按来源分组（合成 gcd vs 真实 ugcd_full），再在来源内按类别进一步划分
- **Non-IID 类型**：复合异构（域偏移 + 类别异构）

#### 策略 4：Non-IID1 / Non-IID2 / IID 类别对比（`partition_noniid.py` + `partition_iid.py`）

专为异构度对比实验设计。三类划分用**相同客户端数 K=70** 和**相近客户端尺寸（均值≈390）**，使**唯一变量是数据异构度**。**类 0（无变化）作为独立的第 7 个语义类**，与 6 个变化类 {2..7} 一视同仁：

| 设置 | 每客户端语义类数 | 每类客户端数 | 文件 |
|------|----------------|-------------|------|
| **Non-IID1** | 1 | 10 | `partition_noniid1_K70.json` |
| **Non-IID2** | 2 | 20 | `partition_noniid2_K70.json` |
| **IID** | 全部 7 类 | 全部 | `partition_iid_K70.json` |

```bash
python partition_noniid.py --classes_per_client 1 --num_clients 70   # Non-IID1
python partition_noniid.py --classes_per_client 2 --num_clients 70   # Non-IID2
python partition_iid.py --num_clients 70                             # IID 基线
```

- **7 个语义类**：`{0 无变化, 2 建筑, 3 道路, 4 水体, 5 荒地, 6 森林, 7 农业}`（类 0 = ucd+ugcd+ugcd_full 负样本）
- **类别分配**采用循环窗口：客户端 `j` 持有连续 `classes_per_client` 个类 `C[(j+t) % 7]`，对任意 `classes_per_client` 都均衡
- **类 0 不再均撒**，而是像其他类一样分给其归属客户端。Non-IID1 会产生约 10 个**纯无变化客户端**（只含类 0），忠实反映极端类异构（某些区域观测期内无变化）
- 与原 FedSeg 公式（K=类别数×每类客户端数，K 随设置变化）不同，本方案固定 K=70，确保 Non-IID1/2/IID 可直接对比
- 一键运行 6 个对比实验：`bash scripts/run_fed_class_comparison.sh`

### 已生成的划分文件

| 文件 | 客户端数 | 总样本 | 异构类型 |
|------|---------|--------|---------|
| `partition_source.json` | 7 | 27,334 | 域偏移 |
| `partition_dirichlet_a0.1_n7.json` | 7 | 27,334 | 极端统计异构 |
| `partition_dirichlet_a0.5_n7.json` | 7 | 27,334 | 中度统计异构 |
| `partition_dirichlet_a1.0_n7.json` | 7 | 27,334 | 轻度统计异构 |
| `partition_dirichlet_a100.0_n7.json` | 7 | 27,334 | 近 IID（基线） |
| `partition_hybrid_c1_separate.json` | 6 | 13,267 | 域偏移 + 类别异构 |
| `partition_noniid1_K70.json` | 70 | 27,334 | 类别异构（1类/客户端，含类0） |
| `partition_noniid2_K70.json` | 70 | 27,334 | 类别异构（2类/客户端，含类0） |
| `partition_iid_K70.json` | 70 | 27,334 | 分层 IID（对比基线） |

### 重新生成划分

```bash
# 查看划分统计（不写文件）
python partition_by_source.py --dry_run

# 生成新参数的划分
python partition_dirichlet.py --alpha 0.3 --num_clients 10
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

### 2. 快速运行联邦实验（2 轮验证）

```bash
python -m fed_cd.federated.fed_main \
    --partition_json partitions/partition_source.json \
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

> **说明**：性能上界由**集中式训练**（`fed_cd.centralized.cen_main`）提供；IID 划分下的联邦训练（`partition_iid_K70.json` + `--iid True` 简单聚合）作为近上界参考，与 Non-IID 各档同框架对比。

### 联邦算法

本项目支持以下联邦学习算法：

| 算法 | 关键参数 | 原理 |
|------|---------|------|
| **FedAvg** | `--fedprox_mu 0.0` | 经典联邦平均，所有客户端权重简单平均 |
| **加权 FedAvg** | `--iid False` | 按客户端数据量加权平均（默认用于 Non-IID） |
| **FedProx** | `--fedprox_mu 0.01` | 在本地损失中加入近端正则项，约束客户端模型不偏离全局模型太远 |
| **全局 EMA** | `--globalema True` | 对全局模型参数使用指数移动平均，提升稳定性 |

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
    --fedprox_mu 0.01 \
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --project_name "FedProx_dirichlet05_bcd"
```

### 批量运行 BCD 实验（12 组）

一次性运行 FedAvg 和 FedProx 在 6 种划分上的全部实验：

```bash
# PowerShell
powershell -ExecutionPolicy Bypass -File scripts\run_fed_bcd.ps1

# Bash
bash scripts/run_fed_bcd.sh
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 全局通信轮数 |
| `--frac_num` | 5 | 每轮参与训练的客户端数量 |
| `--local_ep` | 2 | 每个客户端本地训练的 epoch 数 |
| `--local_bs` | 8 | 本地训练 batch size |
| `--img_size` | 256 | 输入图像尺寸（从 512 缩放） |
| `--lr` | 0.01 | 初始学习率 |
| `--lr_policy` | linear | 学习率衰减策略 |
| `--fedprox_mu` | 0.0 | FedProx 近端项权重（0 = 纯 FedAvg） |
| `--pretrained` | True | 是否使用 ImageNet 预训练 ResNet 权重 |
| `--iid` | False | True=简单平均, False=加权平均 |
| `--globalema` | False | 是否使用全局 EMA |

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
# Bash（FedAvg + FedProx，Dirichlet α=0.5）
bash scripts/run_fed_torchange_bcd.sh

# PowerShell
powershell -ExecutionPolicy Bypass -File scripts\run_fed_torchange_bcd.ps1
```

脚本默认在 `partition_dirichlet_a0.5_n7.json` 上运行；如需遍历多种异构度，取消脚本中 `PARTITIONS` 数组的注释行即可。也可直接指向 IID 划分 `partition_iid_K70.json` 作为各模型的联邦近上界对比；真正的性能上界用 `bash scripts/run_centralized.sh` 跑集中式版本。

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
| 联邦 IID | `partition_iid_K70.json` | FedAvg (`--iid True`) | 近上界参考 |
| 联邦 IID | Dirichlet α=100 | FedAvg | 近 IID 参考 |

#### Phase 2：Non-IID 对比

| 划分 | FedAvg | FedProx (μ=0.01) |
|------|--------|------------------|
| Source (域偏移) | ✓ | ✓ |
| Dirichlet α=0.1 | ✓ | ✓ |
| Dirichlet α=0.5 | ✓ | ✓ |
| Dirichlet α=1.0 | ✓ | ✓ |
| Hybrid | ✓ | ✓ |

#### Phase 3：消融实验

| 消融维度 | 变化范围 | 研究问题 |
|---------|---------|---------|
| Dirichlet α | 0.1, 0.5, 1.0, 100 | 数据异构度的影响 |
| FedProx μ | 0, 0.001, 0.01, 0.1, 1.0 | 近端约束强度 |
| local_ep | 1, 2, 5 | 本地训练强度 |
| frac_num | 1, 5, 10, 20, 全部 | 客户端参与率（`scripts/run_fed_fracnum_sweep.sh`） |
| img_size | 128, 256, 512 | 输入分辨率 |

> **frac_num 稳健性 sweep**：默认在强异构 Non-IID1（K=70）上跑 `[1, 5, 10, 20, 70]`，验证结论对客户端参与率（1.4%~100%）是否稳健：`bash scripts/run_fed_fracnum_sweep.sh`（可自定义 partition 和 frac_num 序列）。

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

#### Phase 5：Non-IID 类别异构对比（FedSeg 风格）

固定 K=70，唯一变量为数据异构度（BCD 任务；集中式上界见 Phase 1）。类 0（无变化）作为第 7 个语义类参与异构：

| 划分 | 每客户端类数 | FedAvg | FedProx (μ=0.01) |
|------|-------------|--------|------------------|
| Non-IID1 | 1（含纯无变化客户端） | ✓ | ✓ |
| Non-IID2 | 2 | ✓ | ✓ |
| IID（分层） | 全部 7 类 | ✓ | ✓ |

> 运行脚本：先生成划分（`partition_noniid.py` ×2 + `partition_iid.py`），再 `bash scripts/run_fed_class_comparison.sh`。共 6 个实验。

### 结果分析维度

| 分析 | 关键指标 |
|------|---------|
| 异构度影响 | Dirichlet α 从 100→0.1 的性能变化；Non-IID1→Non-IID2→IID 的递变 |
| 算法对比 | FedAvg vs FedProx 在不同划分下的差异 |
| 合成 vs 真实 | test vs test2 性能差距 |
| 通信效率 | 达到 90% IID 性能所需通信轮数 |
| 负样本影响 | 负样本客户端对全局模型的贡献 |
| 类别异构程度 | Non-IID1/2/IID 三档对比，每客户端类数 1/2/7 的性能曲线 |

---

## 常见问题

### Q: CUDA 不可用怎么办？

确认安装了 GPU 版本的 PyTorch。CPU 版本可用于代码验证，但训练速度约为 GPU 的 1/50。

### Q: 内存不足（OOM）？

- 减小 `--local_bs`（如 4 或 2）
- 减小 `--img_size`（如 128）
- 使用 `--pretrained False` 跳过下载预训练权重

### Q: 如何添加新的划分策略？

1. 在根目录创建新的 `partition_xxx.py` 脚本
2. 导入 `from partition_utils import *` 获取公共工具
3. 输出标准 JSON 格式（参考已有划分文件结构）
4. 每个样本包含 `im1, im2, label, mask1, mask2, source` 字段

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
