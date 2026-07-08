# Federated algorithm comparison: FedAvg vs FedProx vs FedNova vs SCAFFOLD.
# Core P0 SOTA-comparison experiment on a fixed Non-IID partition, × seeds.
#
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_fed_alg_comparison.ps1

param(
    [string]$Partition = "partitions/partition_dirichlet_a0.5_n7.json",
    [int[]]$Seeds = @(42, 2024, 0),
    [int]$Epochs = 200,
    [int]$FracNum = 5,
    [int]$LocalEp = 2,
    [int]$BatchSize = 8,
    [int]$ImgSize = 256,
    [float]$Lr = 0.01,
    [float]$FedProxMu = 0.01
)

$TAG = [System.IO.Path]::GetFileNameWithoutExtension($Partition) `
    -replace '^partition_(.+)$', '$1'

if (-not (Test-Path $Partition)) {
    Write-Host "缺少 $Partition，请先生成划分文件。" -ForegroundColor Yellow
    exit 1
}

$ALGS = @("fedavg", "fedprox", "fednova", "scaffold")

foreach ($seed in $Seeds) {
    $COMMON = @(
        "--data_root", "../WHU-GCD",
        "--net_G", "base_transformer_pos_s4_dd8",
        "--num_classes", "2",
        "--img_size", $ImgSize,
        "--epochs", $Epochs,
        "--frac_num", $FracNum,
        "--local_ep", $LocalEp,
        "--local_bs", $BatchSize,
        "--lr", $Lr,
        "--lr_policy", "linear",
        "--optimizer", "sgd",
        "--pretrained", "True",
        "--eval_splits", "val,test,test2",
        "--global_test_frequency", "20",
        "--save_frequency", "20",
        "--checkpoint_root", "results/alg_comparison",
        "--partition_json", $Partition,
        "--iid", "False",
        "--seed", "$seed"
    )
    Write-Host "`n========== Seed = $seed (partition=$TAG) ==========" -ForegroundColor Cyan

    foreach ($alg in $ALGS) {
        $proj = "${alg}_${TAG}_bcd_s$seed"
        Write-Host ">>> $proj" -ForegroundColor Yellow
        $a = @("--fed_alg", $alg, "--fedprox_mu", "$FedProxMu",
               "--project_name", $proj) + $COMMON
        python -m fed_cd.federated.fed_main @a
    }
}

Write-Host "`n========== Algorithm comparison complete ==========" -ForegroundColor Green
Write-Host "汇总: python -m fed_cd.summarize --results_root results/alg_comparison"
