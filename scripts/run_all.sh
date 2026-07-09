#!/bin/bash
# =============================================================================
# 总入口：用【一个随机种子】跑完全部对比实验
# =============================================================================
#
# 传一个 seed，本脚本依次调用所有实验子脚本，每个子脚本都用这同一个 seed。
# 这是最常用的"单 seed 全量实验"工作流（跑 3 个 seed 即调用 3 次本脚本）。
#
# Run from FedChange/ directory:
#   bash scripts/run_all.sh                    # 默认 seed=42，跑全部实验
#   bash scripts/run_all.sh 2024               # 指定 seed
#   bash scripts/run_all.sh 0 --epochs 100     # 覆盖训练轮数等公共超参
#
# 跑 3 个 seed 的完整工作流：
#   for s in 42 2024 0; do bash scripts/run_all.sh $s; done
#   python -m fed_cd.summarize                 # 汇总所有 results/，自动产出 mean±std 表
#
# ─── 实验清单（按执行顺序）──────────────────────────────────────────
#   0) env check          环境依赖预检（按将运行的步骤决定查核心库/额外查 torchange 库）
#   1) centralized        集中式上界（BIT-CD，全量数据集中训练）
#   2) alg_comparison     4 联邦算法对比（FedAvg/FedProx/FedNova/SCAFFOLD，Dirichlet a0.5）
#   3) fed_bcd            4 个 Dirichlet α × (FedAvg+FedProx) 异构度主矩阵
#   4) torchange_fed      torchange 基线联邦对比（Dirichlet a0.5）
#   5) changen2           Changen2 零样本评估（免训练）
#
# ─── 选择性执行（--only / --skip，逗号分隔步骤名）─────────────────
#   只跑单个对比试验：
#     bash scripts/run_all.sh 42 --only alg_comparison
#   跑指定的若干个：
#     bash scripts/run_all.sh 42 --only fed_bcd,torchange_fed
#   跳过部分（例如只跑需要训练的、跳过免训练的零样本评估）：
#     bash scripts/run_all.sh 42 --skip centralized,changen2
#   可用步骤名: centralized | alg_comparison | fed_bcd | torchange_fed | changen2
# =============================================================================

set -uo pipefail  # 未定义变量报错 + 管道失败传播；不设 -e 以便单步失败不中断后续实验

# 运行一步子脚本，失败时打印醒目提示（不中断后续步骤，适合长时间批跑）
run_step() {
    local label="$1"; shift
    if "$@"; then
        echo ">>> [OK] $label"
    else
        local rc=$?
        echo ""; echo ">>> [FAIL rc=$rc] $label — 跳过本步，继续后续实验" >&2
        echo ""
    fi
}

SEED=${1:-42}
shift 2>/dev/null || true   # 剩余参数当作公共超参覆盖 / 步骤开关

# ─── 公共超参（可被 --epochs 等覆盖）───
EPOCHS=${EPOCHS:-200}
FRAC_NUM=${FRAC_NUM:-5}
LOCAL_EP=${LOCAL_EP:-2}
BATCH_SIZE=${BATCH_SIZE:-8}
IMG_SIZE=${IMG_SIZE:-256}
LR=${LR:-0.01}
FEDPROX_MU=${FEDPROX_MU:-0.01}

# ─── 步骤开关（默认全部开启）───
ONLY=""; SKIP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --only)  ONLY="$2";  shift 2;;
        --skip)  SKIP="$2";  shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --frac)   FRAC_NUM="$2"; shift 2;;
        --local_ep) LOCAL_EP="$2"; shift 2;;
        --bs)     BATCH_SIZE="$2"; shift 2;;
        --img)    IMG_SIZE="$2"; shift 2;;
        --lr)     LR="$2"; shift 2;;
        --mu)     FEDPROX_MU="$2"; shift 2;;
        *) echo "未知参数: $1"; exit 1;;
    esac
done

# 判断某步骤是否该执行
should_run() {
    local name="$1"
    # --skip 优先：在 skip 列表里就跳过
    if [ -n "$SKIP" ] && echo ",$SKIP," | grep -q ",$name,"; then
        return 1
    fi
    # 若指定了 --only，则只跑 only 列表里的
    if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$name,"; then
        return 1
    fi
    return 0
}

# 子脚本参数模板（统一的公共超参）
COMMON_PASS=(
    "$EPOCHS" "$FRAC_NUM" "$LOCAL_EP" "$BATCH_SIZE" "$IMG_SIZE" "$LR" "$FEDPROX_MU"
)

echo "############################################################"
echo "#  FedChange 全量实验 — seed = $SEED"
echo "#  epochs=$EPOCHS frac=$FRAC_NUM local_ep=$LOCAL_EP bs=$BATCH_SIZE img=$IMG_SIZE lr=$LR mu=$FEDPROX_MU"
[ -n "$ONLY" ] && echo "#  --only: $ONLY"
[ -n "$SKIP" ] && echo "#  --skip: $SKIP"
echo "############################################################"

# ─── 步骤 0：环境依赖预检（只检查本次将要运行的步骤所需库）───
# shellcheck source=check_env.sh
source "$(dirname "$0")/check_env.sh"
# 任一将运行的步骤都需要核心库
check_env_core || exit 1
# torchange 基线步骤（torchange_fed / changen2）额外需要可选库
if should_run torchange_fed || should_run changen2; then
    check_env_torchange || exit 1
fi

# ─── 步骤 1：集中式上界 ───
if should_run centralized; then
    echo ""; echo "======== [1/5] centralized (上界) — seed=$SEED ========"
    run_step "centralized" bash scripts/run_centralized.sh "$EPOCHS" "$BATCH_SIZE" "$IMG_SIZE" "$LR" \
        base_transformer_pos_s4_dd8 base "$SEED"
fi

# ─── 步骤 2：4 联邦算法对比 ───
if should_run alg_comparison; then
    echo ""; echo "======== [2/5] alg_comparison (FedAvg/FedProx/FedNova/SCAFFOLD) — seed=$SEED ========"
    # run_fed_alg_comparison.sh 参数: partition seeds_str epochs frac local_ep bs img lr mu
    run_step "alg_comparison" bash scripts/run_fed_alg_comparison.sh \
        partitions/partition_dirichlet_a0.5_n7.json "$SEED" \
        "$EPOCHS" "$FRAC_NUM" "$LOCAL_EP" "$BATCH_SIZE" "$IMG_SIZE" "$LR" "$FEDPROX_MU"
fi

# ─── 步骤 3：Dirichlet α 异构度主矩阵 ───
if should_run fed_bcd; then
    echo ""; echo "======== [3/5] fed_bcd (4 个 Dirichlet α × FedAvg+FedProx) — seed=$SEED ========"
    # run_fed_bcd.sh 参数: epochs frac local_ep bs img lr mu seeds_str
    run_step "fed_bcd" bash scripts/run_fed_bcd.sh "${COMMON_PASS[@]}" "$SEED"
fi

# ─── 步骤 4：torchange 联邦对比 ───
if should_run torchange_fed; then
    echo ""; echo "======== [4/5] torchange_fed (BIT-CD vs torchange 基线) — seed=$SEED ========"
    # run_fed_torchange_bcd.sh 参数: epochs frac local_ep bs img lr mu seeds_str
    run_step "torchange_fed" bash scripts/run_fed_torchange_bcd.sh "${COMMON_PASS[@]}" "$SEED"
fi

# ─── 步骤 5：Changen2 零样本评估（免训练，与 seed 无关）───
if should_run changen2; then
    echo ""; echo "======== [5/5] changen2_zeroshot (零样本评估) ========"
    run_step "changen2_zeroshot" python scripts/run_changen2_zeroshot.py --data_root ../WHU-GCD
fi

echo ""
echo "############################################################"
echo "#  全量实验完成 — seed = $SEED"
echo "#  汇总全部 seed 的结果（自动按 _s<seed> 分组并产出 mean±std）："
echo "#    python -m fed_cd.summarize"
echo "############################################################"
