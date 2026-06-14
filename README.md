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
- [集中式基线训练](#集中式基线训练)
- [联邦学习训练](#联邦学习训练)
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
- **完整的评估体系**：F1、IoU、Precision、Recall，在合成测试集和真实测试集上分别评估
- **集中式基线**：提供非联邦的上界对比

---

## 项目结构

```
FedChange/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
│
├── partition_utils.py                 # 数据划分公共工具（扫描、解析、统计）
├── partition_by_source.py             # 策略 1：按数据来源划分（域偏移）
├── partition_by_class.py              # 策略 2：按语义类别划分（类异构）
├── partition_dirichlet.py             # 策略 3：Dirichlet 分布划分（可控异构度）
├── partition_hybrid.py                # 策略 4：混合划分（域偏移 + 类异构）
│
├── partitions/                        # 已生成的 Non-IID 划分文件（JSON）
│   ├── partition_source.json
│   ├── partition_class_c1_separate.json
│   ├── partition_class_c2_distribute.json
│   ├── partition_dirichlet_a0.1_n7.json
│   ├── partition_dirichlet_a0.5_n7.json
│   ├── partition_dirichlet_a1.0_n7.json
│   ├── partition_dirichlet_a100.0_n7.json
│   └── partition_hybrid_c1_separate.json
│
├── fed_cd/                            # 核心代码包
│   ├── options.py                     # 统一命令行参数管理
│   ├── centralized_main.py            # 集中式训练入口（基线）
│   ├── summarize.py                   # 实验结果汇总工具
│   │
│   ├── data/                          # 数据处理
│   │   ├── cd_dataset.py              #   WHU-GCD 双时相数据集（BCD/SCD）
│   │   ├── data_partition.py          #   划分加载 + 评估集扫描
│   │   └── data_utils.py              #   同步双时相数据增强
│   │
│   ├── models/                        # 模型
│   │   └── bit_cd.py                  #   BIT-CD 模型（ResNet18 + Transformer）
│   │
│   ├── federated/                     # 联邦学习
│   │   ├── fed_main.py                #   联邦主循环
│   │   ├── local_update.py            #   本地训练（含 FedProx）
│   │   └── aggregation.py             #   权重聚合 + EMA
│   │
│   └── evaluation/                    # 评估
│       ├── cd_metrics.py              #   混淆矩阵指标计算
│       └── evaluator.py               #   多集评估器
│
├── scripts/                           # 运行脚本
│   ├── run_centralized.ps1            # 集中式基线（PowerShell）
│   ├── run_centralized.sh             # 集中式基线（Bash）
│   ├── run_fed_bcd.ps1                # BCD 联邦实验（PowerShell）
│   └── run_fed_bcd.sh                 # BCD 联邦实验（Bash）
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

# 集中式基线
python -m fed_cd.centralized_main --data_root /home/user/datasets/WHU-GCD ...

# 联邦实验
python -m fed_cd.federated.fed_main --data_root /home/user/datasets/WHU-GCD ...
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

#### 策略 2：按语义类别划分（`partition_by_class.py`）

借鉴 FedSeg 的逐类划分策略，按变化涉及的语义类别分配数据。

```bash
# 每客户端持有 1 个变化类别，负样本单独分配
python partition_by_class.py --classes_per_client 1 --neg_strategy separate

# 每客户端持有 2 个变化类别，负样本均匀分散
python partition_by_class.py --classes_per_client 2 --neg_strategy distribute
```

- **模拟场景**：不同客户端只观测到特定地物类型的变化（如沿海客户端主要观测水体变化）
- **Non-IID 类型**：类别异构（Class-Heterogeneous）

#### 策略 3：Dirichlet 分布划分（`partition_dirichlet.py`）

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

#### 策略 4：混合划分（`partition_hybrid.py`）

同时模拟域偏移和类别异构两种 Non-IID 场景。

```bash
python partition_hybrid.py --domain_clients 3,2 --classes_per_client 1
```

- 先将数据按来源分组（合成 gcd vs 真实 ugcd_full），再在来源内按类别进一步划分
- **Non-IID 类型**：复合异构（域偏移 + 类别异构）

### 已生成的划分文件

| 文件 | 客户端数 | 总样本 | 异构类型 |
|------|---------|--------|---------|
| `partition_source.json` | 7 | 27,334 | 域偏移 |
| `partition_class_c1_separate.json` | 7 | 27,334 | 类别异构（1类/客户端） |
| `partition_class_c2_distribute.json` | 3 | 27,334 | 类别异构（2类/客户端） |
| `partition_dirichlet_a0.1_n7.json` | 7 | 27,334 | 极端统计异构 |
| `partition_dirichlet_a0.5_n7.json` | 7 | 27,334 | 中度统计异构 |
| `partition_dirichlet_a1.0_n7.json` | 7 | 27,334 | 轻度统计异构 |
| `partition_dirichlet_a100.0_n7.json` | 7 | 27,334 | 近 IID（基线） |
| `partition_hybrid_c1_separate.json` | 6 | 13,267 | 域偏移 + 类别异构 |

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

## 集中式基线训练

集中式训练使用全部数据（不分客户端）训练 BIT-CD 模型，作为联邦学习的性能上界。

### 运行方式

```bash
# PowerShell
powershell -ExecutionPolicy Bypass -File scripts\run_centralized.ps1

# Bash
bash scripts/run_centralized.sh

# 或直接运行（自定义参数）
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
    --pretrained True \
    --eval_splits "val,test,test2" \
    --global_test_frequency 20 \
    --project_name "Centralized_A"
```

### 两种采样策略

遵循 WHU-GCD 数据集论文的训练建议：

| 策略 | 数据组合 | 配比 | 适用场景 |
|------|---------|------|---------|
| 策略 A | gcd + ugcd_full | 3:1 | 侧重真实数据，推荐 |
| 策略 B | gcd + ucd + ugcd | 6:1:1 | 更多负样本，平衡 |

---

## 联邦学习训练

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

### 批量运行 BCD 实验（14 组）

一次性运行 FedAvg 和 FedProx 在 7 种划分上的全部实验：

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

## 评估与结果汇总

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

输出示例：

```
Found 3 experiment(s) in results/

Experiment                               |          val           |          test          |         test2
                                         |     mF1    mIoU    F1_c |     mF1    mIoU    F1_c |     mF1    mIoU    F1_c
=============================================================================================================================
Centralized_A                            |  89.32%  82.15%  85.67% |  87.45%  80.12%  83.23% |  76.89%  68.34%  71.56%
FedAvg_source_bcd                        |  85.67%  77.89%  81.23% |  83.12%  75.45%  78.90% |  72.45%  63.12%  66.78%
FedProx0.01_dirichlet05_bcd              |  84.23%  76.34%  79.56% |  82.01%  74.12%  77.45% |  71.23%  62.01%  65.34%
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
| 集中式 A | 全部数据（3:1） | — | 性能上界 |
| 集中式 B | 全部数据（6:1:1） | — | 性能上界 |
| 联邦 IID | Dirichlet α=100 | FedAvg | 联邦学习上界 |

#### Phase 2：Non-IID 对比

| 划分 | FedAvg | FedProx (μ=0.01) |
|------|--------|------------------|
| Source (域偏移) | ✓ | ✓ |
| Dirichlet α=0.1 | ✓ | ✓ |
| Dirichlet α=0.5 | ✓ | ✓ |
| Dirichlet α=1.0 | ✓ | ✓ |
| Class c1 | ✓ | ✓ |
| Hybrid | ✓ | ✓ |

#### Phase 3：消融实验

| 消融维度 | 变化范围 | 研究问题 |
|---------|---------|---------|
| Dirichlet α | 0.1, 0.5, 1.0, 100 | 数据异构度的影响 |
| FedProx μ | 0, 0.001, 0.01, 0.1, 1.0 | 近端约束强度 |
| local_ep | 1, 2, 5 | 本地训练强度 |
| frac_num | 2, 5, 全部 | 客户端参与率 |
| img_size | 128, 256, 512 | 输入分辨率 |

### 结果分析维度

| 分析 | 关键指标 |
|------|---------|
| 异构度影响 | Dirichlet α 从 100→0.1 的性能变化 |
| 算法对比 | FedAvg vs FedProx 在不同划分下的差异 |
| 合成 vs 真实 | test vs test2 性能差距 |
| 通信效率 | 达到 90% 集中式性能所需通信轮数 |
| 负样本影响 | 负样本客户端对全局模型的贡献 |

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

替换 `fed_cd/models/bit_cd.py` 中的模型，只需保证：
- `forward(x1, x2)` 接收双时相输入
- 输出 logits 的 shape 为 `(B, num_classes, H, W)`
- 使用 `build_bit_cd_model()` 工厂函数返回模型实例

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

WHU-GCD 数据集相关问题请联系：yujiezan@whu.edu.cn

---

## License

本项目代码仅供非商业和研究目的使用。
