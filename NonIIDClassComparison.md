# Non-IID 类别对比实验：实施记录

> 本文档记录为 FedChange 框架新增 **Non-IID1 / Non-IID2 / IID 类别异构对比实验** 的完整过程：需求分析、调研、设计决策、算法实现、验证与使用方法。
>
> 参考：FedSeg (*Class-Heterogeneous Federated Learning for Semantic Segmentation*, CVPR 2023) 的类别异构思想（见 `FedSegNonIID.md`）。

---

## 目录

- [1. 需求](#1-需求)
- [2. 调研过程](#2-调研过程)
- [3. 关键设计决策](#3-关键设计决策)
- [4. 算法设计](#4-算法设计)
- [5. 代码实现](#5-代码实现)
- [6. 验证结果](#6-验证结果)
- [7. 使用方法](#7-使用方法)
- [8. 实验矩阵](#8-实验矩阵)
- [9. 与原 FedSeg 的差异](#9-与原-fedseg-的差异)
- [10. 迭代记录：类 0 从"均撒"改为"独立语义类"](#10-迭代记录类-0-从均撒改为独立语义类)

---

## 1. 需求

在 FedChange（联邦遥感图像变化检测）框架中，新增一组**按语义类别划分的 Non-IID 对比实验**，用于研究数据异构度对联邦变化检测性能的影响：

- **Non-IID1**：每个客户端只持有 **1 个**语义类
- **Non-IID2**：每个客户端持有 **2 个**语义类
- **IID**：作为对比基线，需要 IID 数据
- **不需要 centralized 训练**
- 数据加载方式参考 FedSeg（类别异构，非 Dirichlet）
- 训练任务为**二值变化检测 BCD**（`num_classes=2`）
- **类 0（无变化）作为独立的语义类**参与划分（与 6 个变化类一视同仁）

WHU-GCD 共 **7 个语义类**：`{0 无变化, 2 建筑, 3 道路, 4 水体, 5 荒地, 6 森林, 7 农业}`。类 0 = 负样本（ucd/ugcd/ugcd_full）。

---

## 2. 调研过程

### 2.1 现状梳理

| 项目 | 现状 | 结论 |
|------|------|------|
| 类别异构划分 | 旧 `partition_by_class.py`（FedSeg 风格 70 客户端、负样本单独处理）已弃用并删除 | 由 `partition_noniid.py`（固定 K + 类 0 作同类）取代 |
| IID 基线 | 仅 Dirichlet α=100（近 IID，随机非分层） | 需新增干净分层 IID |
| centralized | 项目已移除集中式训练（性能上界改由 IID 划分充当） | 无需处理 |

### 2.2 关键约束发现

`fed_cd/federated/fed_main.py:128-129`：

```python
num_clients = len(client_ids)   # 客户端数取自 partition JSON
frac_num = min(args.frac_num, num_clients)
```

**客户端数由 partition JSON 决定**（`--num_users` 形同虚设），`--frac_num` 上限为该值。因此 Non-IID1 / Non-IID2 / IID 三者**必须使用相同客户端数 K**，否则对比被客户端数混淆。

### 2.3 真实类别分布（实测）

| 类 ID | 类别 | 样本数 | 来源 |
|-------|------|--------|------|
| **0** | **无变化** | **6,334** | ucd+ugcd+ugcd_full（作为独立语义类） |
| 2 | 建筑 | 3,949 | gcd |
| 3 | 道路 | 1,517 | gcd |
| 4 | 水体 | 3,338 | gcd |
| 5 | 荒地 | 2,078 | gcd |
| 6 | 森林 | 5,676 | gcd |
| 7 | 农业 | 4,442 | gcd |
| — | **总计** | **27,334** | |

各类样本数**不均衡**（类 0 6334、森林 5676 vs 道路 1517），导致类别异构划分的客户端尺寸有方差——属固有特性，非 bug。

> **类 0 纯度说明**：类 0 组 = ucd + ugcd（纯无变化）+ ugcd_full（主要无变化，但按 WHU-GCD 规范含少量真实变化）。这与现有 Dirichlet 划分对这些来源的标记方式一致。

---

## 3. 关键设计决策

### 3.1 类 0 作为独立语义类（本次迭代的核心）

**早期方案**把负样本"均撒"到所有客户端（`distribute_equal`），不参与类别异构结构。**现方案**把类 0 作为第 7 个语义类，与变化类一同进入循环窗口——类 0 不再特殊处理，而是像其他类一样分给其归属客户端。

**副作用（已确认接受）**：Non-IID1 会产生约 10 个**纯无变化客户端**（只含类 0）。这些客户端的本地模型只学到"无变化"，聚合时可能与变化类客户端冲突。这是忠实反映极端类异构（某些区域观测期内确实无变化）的合法场景，是研究的意义所在，非缺陷。

### 3.2 客户端数 K=70

7 个语义类，要求"每类 10 客户端" → K = 7 × 10 = **70**（60 不能被 7 整除）。三设置（Non-IID1/2/IID）统一 K=70：

| 设置 | K | 每客户端类数 | 每类客户端数 |
|------|----|----|----|
| Non-IID1 | 70 | 1 | 恰好 10 |
| Non-IID2 | 70 | 2 | 20 |
| IID | 70 | 全部 7 类 | 全部 |

### 3.3 聚合方式（遵循 FedSeg 惯例）

- Non-IID：加权聚合 `--iid False`（按客户端数据量加权）
- IID：简单聚合 `--iid True`

---

## 4. 算法设计

### 4.1 循环窗口类别分配（核心）

将 7 个语义类 `C = [0, 2, 3, 4, 5, 6, 7]` 排成环，客户端 `j` 持有连续 `classes_per_client` 个类：

```
assigned_classes[j] = [ C[(j + t) % 7]  for t in range(classes_per_client) ]
```

**均衡性证明**：类 `C[i]` 出现在客户端 `j` 当且仅当 `j ≡ i, i-1, ..., i-(n-1) (mod 7)`，即 `n` 个余数类，每类 `K/7 = 10` 个客户端 → 每类恰好 `n × 10` 个客户端。

**示例**（K=70）：

| 客户端 | Non-IID1 (n=1) | Non-IID2 (n=2) |
|--------|----------------|----------------|
| client_0 | {0} ← 纯无变化 | {0, 2} |
| client_1 | {2} | {2, 3} |
| client_2 | {3} | {3, 4} |
| client_3 | {4} | {4, 5} |
| client_4 | {5} | {5, 6} |
| client_5 | {6} | {6, 7} |
| client_6 | {7} | {7, 0} |
| client_7 | {0} ← 纯无变化 | {0, 2} |
| ... | ...（每 7 个一轮，共 10 轮） | ...（每 7 个一轮，共 10 轮） |

### 4.2 每类样本分配（类 0 与变化类一视同仁）

对每个类 `c`（含类 0），收集所有持有该类的客户端 `holders`，用 `split_list`（shuffle + divmod）将该类样本均分：

```python
# 类 0 = 负样本，与变化类同样处理
class_groups[0] = neg_samples
available_classes = sorted(class_groups.keys())  # [0, 2, 3, 4, 5, 6, 7]

for c in available_classes:
    holders = holders_per_class[c]
    chunks = split_list(class_groups[c], len(holders), seed=seed+c)
    for j, chunk in zip(holders, chunks):
        client_data[f"client_{j}"].extend(chunk)
```

### 4.3 分层 IID

对每个类（含类 0）独立 shuffle 后均分成 K 份，每客户端从每类取一份 → 每客户端含全部 7 类。

---

## 5. 代码实现

### 5.1 新增/修改文件

| 文件 | 作用 |
|------|------|
| `partition_noniid.py` | Non-IID1/2 类别异构划分（**类 0 作第 7 类**，`--classes_per_client 1/2`，固定 K=70，循环窗口） |
| `partition_iid.py` | 分层随机 IID（每客户端含全部 7 类） |
| `partition_utils.py` | `CLASS_NAMES` 加 `0: 无变化`；`compute_client_stats` 把负样本由 `'unknown'` 改标为 `'0'` |
| `scripts/run_fed_class_comparison.sh` | 6 实验批量运行（Bash） |
| `scripts/run_fed_class_comparison.ps1` | 6 实验批量运行（PowerShell） |

### 5.2 CLI 接口

**Non-IID 划分：**
```bash
python partition_noniid.py \
    --classes_per_client {1|2}   # 1=Non-IID1, 2=Non-IID2
    --num_clients 70             # K，三设置必须一致（7 的倍数）
    --data_root ../WHU-GCD
```

**IID 划分：**
```bash
python partition_iid.py --num_clients 70 --data_root ../WHU-GCD
```

### 5.3 输出文件命名

| 设置 | 文件名 |
|------|--------|
| Non-IID1 | `partition_noniid1_K70.json` |
| Non-IID2 | `partition_noniid2_K70.json` |
| IID | `partition_iid_K70.json` |

JSON schema 与现有 partition 完全一致，路径为相对于 `data_root` 的相对路径，跨机器可移植。stats 中负样本显示为 `class_dist['0']`（不再 `'unknown'`）。

### 5.4 运行脚本逻辑

`run_fed_class_comparison.sh` 对 3 个划分各跑 FedAvg（`--fedprox_mu 0.0`）+ FedProx（`--fedprox_mu 0.01`），共 6 实验：

- Non-IID 行用 `--iid False`（加权聚合）
- IID 行用 `--iid True`（简单聚合）
- 固定 `--frac_num 5`（70 选 5，约 7% 参与率）
- 默认 BIT-CD 模型、BCD、`--epochs 200`、`--seed 42`

---

## 6. 验证结果

### 6.1 划分正确性（实测）

| 校验项 | Non-IID1 | Non-IID2 | IID |
|--------|----------|----------|-----|
| 客户端数 K | 70 ✓ | 70 ✓ | 70 ✓ |
| 每类客户端数（含类 0） | 恰好 10 ✓ | 恰好 20 ✓ | 全部 70 ✓ |
| 纯类 0 客户端数 | **10** ✓ | 0 | 0 |
| holders per class | `{0:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10}` ✓ | 各 20 ✓ | 各 70 ✓ |
| stats 含 `'unknown'` | False ✓ | False ✓ | False ✓ |
| 客户端尺寸 min/mean/max | 151 / 390 / 634 | 241 / 390 / 540 | 387 / 390 / 394 |
| 总样本 | 27,334 ✓ | 27,334 ✓ | 27,334 ✓ |

**关键**：三设置**均值尺寸均为 390** → 唯一变量是异构度。Non-IID1 尺寸方差最大（类 0 客户端 ~633、道路客户端 ~151），加权聚合按数据量归一。

### 6.2 端到端 round-trip（模拟 fed_main 加载）

```
partition_noniid1_K70.json: K=70 size min/mean/max=151/390/634  A=(3,256,256) L=[0,1]
partition_noniid2_K70.json: K=70 size min/mean/max=241/390/540  A=(3,256,256) L=[0,1]
partition_iid_K70.json:     K=70 size min/mean/max=387/390/394  A=(3,256,256) L=[0,1]
ALL OK
```

样本 `A=(3,256,256)`、标签 `L={0,1}`（BCD 二值），`fed_main` 加载器成功读取全部三类。

### 6.3 其他校验

- bash 脚本语法 `bash -n` 通过
- 文件名与运行脚本引用完全一致
- CLI argparse 全部分支可解析（`--classes_per_client` choices 含 1–7）

---

## 7. 使用方法

### 7.1 生成三类划分（首次运行前执行一次）

```bash
cd FedChange

# Non-IID1: 每客户端 1 类（含纯无变化客户端）
python partition_noniid.py --classes_per_client 1 --num_clients 70 --data_root ../WHU-GCD

# Non-IID2: 每客户端 2 类
python partition_noniid.py --classes_per_client 2 --num_clients 70 --data_root ../WHU-GCD

# IID: 分层随机
python partition_iid.py --num_clients 70 --data_root ../WHU-GCD
```

> 预览不写文件加 `--dry_run`。生成文件落在 `partitions/`（gitignored，约 8MB/个）。

### 7.2 运行 6 个对比实验

```bash
# 前台
bash scripts/run_fed_class_comparison.sh

# 后台 + 日志（推荐长时间实验）
nohup bash scripts/run_fed_class_comparison.sh > class_comparison.log 2>&1 &
tail -f class_comparison.log
```

PowerShell：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_fed_class_comparison.ps1
```

位置参数自定义（顺序：epochs frac_num local_ep bs img_size lr fedprox_mu K）：
```bash
bash scripts/run_fed_class_comparison.sh 200 5 2 8 256 0.01 0.01 70
```

### 7.3 汇总结果

```bash
python -m fed_cd.summarize --results_root results/class_comparison
```

---

## 8. 实验矩阵

固定 K=70、BCD、无 centralized，唯一变量为数据异构度：

| | Non-IID1（1类/客户端，含纯类0） | Non-IID2（2类/客户端） | IID（分层） |
|---|---|---|---|
| **FedAvg** (`--fedprox_mu 0.0`) | ✓ | ✓ | ✓ |
| **FedProx** (`--fedprox_mu 0.01`) | ✓ | ✓ | ✓ |

**共 6 个实验**。预期趋势：IID ≥ Non-IID2 ≥ Non-IID1（异构度越高性能越低），FedProx 在强异构下应优于 FedAvg。Non-IID1 中的纯无变化客户端会让该档最具挑战性。

---

## 9. 与原 FedSeg 的差异

| 维度 | 原 FedSeg | 本实现 |
|------|-----------|--------|
| 划分依据 | 类别异构（每类独立子文件夹 + "erase"） | 类别异构（按 im2 文件名后缀解析类；类 0 = 负样本） |
| 客户端数 K | `K = 类别数 × 每类客户端数`（随设置变化） | **固定 K=70**（三设置一致，公平对比） |
| 语义类数 | 仅变化类 | **7 类（含类 0 无变化）** |
| 每客户端类数 | 恒为 1（极端 Non-IID） | **1 或 2**（Non-IID1 / Non-IID2） |
| 类别分配 | `client_index = i + class_idx × cpc` | 循环窗口 `C[(j+t) % 7]`（支持任意 classes_per_client） |
| 负样本 | 无（语义分割无纯负样本概念） | **类 0 作为独立语义类**参与划分（Non-IID1 产生纯无变化客户端） |
| IID 基线 | 简单平均 | 分层随机（同 K=70） |
| 聚合 | Non-IID 加权 / IID 简单 | 同（遵循 FedSeg 惯例） |

**核心改进**：固定 K + 循环窗口 + 类 0 入类，使 Non-IID1/Non-IID2/IID 三档可在**同一客户端数和相近客户端尺寸**下对比，把异构度孤立为唯一变量——这是原 FedSeg（K 随设置变化）做不到的。

---

## 10. 迭代记录：类 0 从"均撒"改为"独立语义类"

**v1（初始实现）**：6 个变化类参与循环窗口，负样本（类 0）用 `distribute_equal` 均撒到全部 60 客户端，每客户端 ~105 张。K=60。

**问题**：类 0 未参与类别异构结构，仅作为"上下文补充"均撒——不符合"类 0 作为单独语义类"的诉求。

**v2（当前实现）**：
- 类 0（负样本）并入 `class_groups[0]`，类别池 `[2..7]` → `[0,2,3,4,5,6,7]`（7 类）
- 循环窗口、每类均分对 7 类一视同仁，**删除** `distribute_equal/exclude` 分支
- K 60 → **70**（60 不能被 7 整除；7×10=70 保持"每类 10 客户端"）
- `compute_client_stats` 把负样本由 `'unknown'` 改标 `'0'`，`CLASS_NAMES` 加 `0: 无变化`
- Non-IID1 产生 10 个纯无变化客户端（忠实极端异构）
- 输出文件名去掉 `_distribute_equal` 后缀：`partition_noniid{n}_K{K}.json`

**文件名对照**：

| 旧 (v1, K=60) | 新 (v2, K=70) |
|---|---|
| `partition_noniid1_K60_distribute_equal.json` | `partition_noniid1_K70.json` |
| `partition_noniid2_K60_distribute_equal.json` | `partition_noniid2_K70.json` |
| `partition_iid_K60.json` | `partition_iid_K70.json` |

---

## 附：文件清单

```
FedChange/
├── partition_noniid.py                         # Non-IID1/2 划分（含类 0，K=70）
├── partition_iid.py                            # 分层 IID（K=70）
├── partition_utils.py                          # CLASS_NAMES + 类 0 标签修正
├── scripts/
│   ├── run_fed_class_comparison.sh             # 6 实验批量（Bash）
│   └── run_fed_class_comparison.ps1            # 6 实验批量（PowerShell）
├── partitions/
│   ├── partition_noniid1_K70.json              # 生成（gitignored）
│   ├── partition_noniid2_K70.json              # 生成（gitignored）
│   └── partition_iid_K70.json                  # 生成（gitignored）
├── README.md                                   # 策略 5 + Phase 5 + 划分表
├── DEPLOYMENT.md                               # 第 8 步 + 实验矩阵
├── partitions/README.md                        # Non-IID1/2/IID 表格 + 算法说明
└── NonIIDClassComparison.md                    # 本文档
```
