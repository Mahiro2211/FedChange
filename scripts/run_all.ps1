# =============================================================================
# 总入口：用【一个随机种子】跑完全部对比实验 (PowerShell 版)
#
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1                 # seed=42
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Seed 2024      # 指定 seed
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Seed 0 -Only alg_comparison
#
# 跑 3 个 seed 的完整工作流：
#   foreach ($s in 42,2024,0) { powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Seed $s }
#   python -m fed_cd.summarize   # 汇总，自动产出 mean±std 表
# =============================================================================

param(
    [int]$Seed = 42,
    [int]$Epochs = 200,
    [int]$FracNum = 5,
    [int]$LocalEp = 2,
    [int]$BatchSize = 8,
    [int]$ImgSize = 256,
    [float]$Lr = 0.01,
    [float]$FedProxMu = 0.01,
    # 步骤开关：逗号分隔。例如 "centralized,alg_comparison"
    [string]$Only = "",
    [string]$Skip = ""
)

$ErrorActionPreference = "Stop"

function Should-Run([string]$name) {
    if ($Skip -and (",$Skip," -match ",$name,")) { return $false }
    if ($Only -and -not (",$Only," -match ",$name,")) { return $false }
    return $true
}

Write-Host "############################################################" -ForegroundColor Cyan
Write-Host "#  FedChange 全量实验 — seed = $Seed"
Write-Host "#  epochs=$Epochs frac=$FracNum local_ep=$LocalEp bs=$BatchSize img=$ImgSize lr=$Lr mu=$FedProxMu"
if ($Only)  { Write-Host "#  -Only: $Only" }
if ($Skip)  { Write-Host "#  -Skip: $Skip" }
Write-Host "############################################################" -ForegroundColor Cyan

# ─── 步骤 1：集中式上界 ───
if (Should-Run "centralized") {
    Write-Host "`n======== [1/7] centralized (上界) — seed=$Seed ========" -ForegroundColor Cyan
    # run_centralized.ps1 参数: Epochs BatchSize ImgSize Lr NetG Tag Seed
    & powershell -ExecutionPolicy Bypass -File scripts\run_centralized.ps1 `
        -Epochs $Epochs -BatchSize $BatchSize -ImgSize $ImgSize -Lr $Lr `
        -NetG base_transformer_pos_s4_dd8 -Tag base -Seed $Seed
}

# ─── 步骤 2：4 联邦算法对比 ───
if (Should-Run "alg_comparison") {
    Write-Host "`n======== [2/7] alg_comparison (FedAvg/FedProx/FedNova/SCAFFOLD) — seed=$Seed ========" -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File scripts\run_fed_alg_comparison.ps1 `
        -Partition partitions/partition_dirichlet_a0.5_n7.json `
        -Seeds @($Seed) -Epochs $Epochs -FracNum $FracNum -LocalEp $LocalEp `
        -BatchSize $BatchSize -ImgSize $ImgSize -Lr $Lr -FedProxMu $FedProxMu
}

# ─── 步骤 3：6 划分主矩阵 ───
if (Should-Run "fed_bcd") {
    Write-Host "`n======== [3/7] fed_bcd (6 划分 × FedAvg+FedProx) — seed=$Seed ========" -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File scripts\run_fed_bcd.ps1 `
        -Epochs $Epochs -FracNum $FracNum -LocalEp $LocalEp -BatchSize $BatchSize `
        -ImgSize $ImgSize -Lr $Lr -FedProxMu $FedProxMu -Seeds @($Seed)
}

# ─── 步骤 4：Non-IID 类别异构对比 ───
if (Should-Run "class_comparison") {
    Write-Host "`n======== [4/7] class_comparison (Non-IID1/2/IID, K=70) — seed=$Seed ========" -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File scripts\run_fed_class_comparison.ps1 `
        -Epochs $Epochs -FracNum $FracNum -LocalEp $LocalEp -BatchSize $BatchSize `
        -ImgSize $ImgSize -Lr $Lr -FedProxMu $FedProxMu -K 70 -Seeds @($Seed)
}

# ─── 步骤 5：torchange 联邦对比 ───
if (Should-Run "torchange_fed") {
    Write-Host "`n======== [5/7] torchange_fed (BIT-CD vs torchange 基线) — seed=$Seed ========" -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File scripts\run_fed_torchange_bcd.ps1 `
        -Epochs $Epochs -FracNum $FracNum -LocalEp $LocalEp -BatchSize $BatchSize `
        -ImgSize $ImgSize -Lr $Lr -FedProxMu $FedProxMu -Seeds @($Seed)
}

# ─── 步骤 6：frac_num 参与率稳健性 sweep ───
if (Should-Run "fracnum_sweep") {
    Write-Host "`n======== [6/7] fracnum_sweep (参与率 1.4%~100%) — seed=$Seed ========" -ForegroundColor Cyan
    # run_fed_fracnum_sweep.ps1 参数见该脚本顶部
    & powershell -ExecutionPolicy Bypass -File scripts\run_fed_fracnum_sweep.ps1 `
        -Partition partitions/partition_noniid1_K70.json `
        -FracList 1,5,10,20,70 -Epochs $Epochs -LocalEp $LocalEp `
        -BatchSize $BatchSize -ImgSize $ImgSize -Lr $Lr -Seed $Seed
}

# ─── 步骤 7：Changen2 零样本评估（免训练，与 seed 无关）───
if (Should-Run "changen2") {
    Write-Host "`n======== [7/7] changen2_zeroshot (零样本评估) ========" -ForegroundColor Cyan
    try {
        python scripts/run_changen2_zeroshot.py --data_root ../WHU-GCD
    } catch {
        Write-Host "⚠️  changen2_zeroshot 失败或未安装 torchange，已跳过" -ForegroundColor Yellow
    }
}

Write-Host "`n############################################################" -ForegroundColor Cyan
Write-Host "#  全量实验完成 — seed = $Seed"
Write-Host "#  汇总全部 seed 的结果（自动按 _s<seed> 分组并产出 mean±std）："
Write-Host "#    python -m fed_cd.summarize"
Write-Host "############################################################" -ForegroundColor Cyan
