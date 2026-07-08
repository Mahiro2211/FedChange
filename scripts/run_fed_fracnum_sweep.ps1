# frac_num (客户端参与率) 稳健性 sweep 实验
#
# 在同一 partition 上 sweep 多个 frac_num 值，验证结论对参与率的稳健性
# (弱点 4)。默认在强异构 Non-IID1 (K=70) 上 sweep。仅跑 FedAvg。
#
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_fed_fracnum_sweep.ps1

param(
    [string]$Partition = "partitions/partition_noniid1_K70.json",
    [int[]]$FracList = @(1, 5, 10, 20, 70),
    [int]$Epochs = 200,
    [int]$LocalEp = 2,
    [int]$BatchSize = 8,
    [int]$ImgSize = 256,
    [float]$Lr = 0.01,
    [int]$Seed = 42
)

# 从 partition 文件名解析 tag
$PartitionFile = [System.IO.Path]::GetFileNameWithoutExtension($Partition)
$Tag = $PartitionFile -replace "^partition_", ""
$Tag = $Tag -replace "_K(\d+)$", "_K`$1"

# 读取客户端数 K
$K = 70
if (Test-Path $Partition) {
    $json = Get-Content $Partition -Raw | ConvertFrom-Json
    $K = $json.num_clients
}

if (-not (Test-Path $Partition)) {
    Write-Host "Missing $Partition. Generate it first, e.g.:" -ForegroundColor Red
    Write-Host "  python partition_noniid.py --classes_per_client 1 --num_clients 70"
    exit 1
}

Write-Host "Partition: $Partition (K=$K, tag=$Tag)" -ForegroundColor Cyan
Write-Host "frac_num sweep values: $($FracList -join ', ')"
Write-Host ""

$COMMON = @(
    "--data_root", "../WHU-GCD",
    "--net_G", "base_transformer_pos_s4_dd8",
    "--num_classes", "2",
    "--img_size", $ImgSize,
    "--epochs", $Epochs,
    "--local_ep", $LocalEp,
    "--local_bs", $BatchSize,
    "--lr", $Lr,
    "--lr_policy", "linear",
    "--optimizer", "sgd",
    "--pretrained", "True",
    "--eval_splits", "val,test,test2",
    "--global_test_frequency", "20",
    "--save_frequency", "20",
    "--checkpoint_root", "results/fracnum_sweep",
    "--iid", "False",
    "--seed", "$Seed"
)

Write-Host "========== frac_num sweep on $Tag (K=$K) ==========" -ForegroundColor Cyan
foreach ($frac in $FracList) {
    $pct = if ($frac -ge $K) { "100.0" } else { "{0:N1}" -f ($frac / $K * 100) }
    $proj = "FedAvg_frac${frac}_${Tag}_bcd_s${Seed}"
    Write-Host "`n>>> $proj  (frac_num=$frac, 参与率 ${pct}%)" -ForegroundColor Yellow
    $args = @(
        "--partition_json", $Partition,
        "--project_name", $proj,
        "--fedprox_mu", "0.0",
        "--frac_num", $frac
    ) + $COMMON
    python -m fed_cd.federated.fed_main @args
}

Write-Host "`n========== frac_num sweep 完成 ==========" -ForegroundColor Green
Write-Host "汇总: python -m fed_cd.summarize --results_root results/fracnum_sweep"
