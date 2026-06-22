# Non-IID 类别异构对比实验（FedSeg 风格扩展）：Non-IID1 / Non-IID2 / IID
#
# 三类划分用相同客户端数 K=70、相近客户端尺寸（均值≈390），唯一变量是数据异构度。
# 每类划分 × (FedAvg + FedProx) = 6 个实验。无 centralized。
#
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_fed_class_comparison.ps1
# 生成划分文件需先运行（见脚本末尾注释）。

param(
    [int]$Epochs = 200,
    [int]$FracNum = 5,
    [int]$LocalEp = 2,
    [int]$BatchSize = 8,
    [int]$ImgSize = 256,
    [float]$Lr = 0.01,
    [float]$FedProxMu = 0.01,
    [int]$K = 70
)

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
    "--checkpoint_root", "results/class_comparison",
    "--seed", "42"
)

# 划分: tag | json | --iid (IID=简单平均 True, Non-IID=加权平均 False, 遵循 FedSeg 惯例)
$PARTITIONS = @(
    @{tag="noniid1"; file="partitions/partition_noniid1_K${K}.json"; iid="False"},
    @{tag="noniid2"; file="partitions/partition_noniid2_K${K}.json"; iid="False"},
    @{tag="iid";     file="partitions/partition_iid_K${K}.json";     iid="True"}
)

foreach ($p in $PARTITIONS) {
    if (-not (Test-Path $p.file)) {
        Write-Host "缺少 $($p.file)，请先生成划分（见脚本末注释）。跳过 $($p.tag)。" -ForegroundColor Yellow
        continue
    }

    $projAvg = "FedAvg_$($p.tag)_K${K}_bcd"
    Write-Host "`n>>> $projAvg" -ForegroundColor Cyan
    $a = @("--partition_json", $p.file, "--project_name", $projAvg,
           "--fedprox_mu", "0.0", "--iid", $p.iid) + $COMMON
    python -m fed_cd.federated.fed_main @a

    $projProx = "FedProx${FedProxMu}_$($p.tag)_K${K}_bcd"
    Write-Host "`n>>> $projProx" -ForegroundColor Cyan
    $a = @("--partition_json", $p.file, "--project_name", $projProx,
           "--fedprox_mu", $FedProxMu, "--iid", $p.iid) + $COMMON
    python -m fed_cd.federated.fed_main @a
}

Write-Host "`n========== Non-IID 类别对比实验完成 ==========" -ForegroundColor Green

# ─── 生成划分文件（首次运行前执行一次）───
# python partition_noniid.py --classes_per_client 1 --num_clients 70
# python partition_noniid.py --classes_per_client 2 --num_clients 70
# python partition_iid.py --num_clients 70
