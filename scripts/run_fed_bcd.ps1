# Federated BCD experiments: FedAvg and FedProx on multiple Non-IID partitions
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_fed_bcd.ps1

param(
    [int]$Epochs = 200,
    [int]$FracNum = 5,
    [int]$LocalEp = 2,
    [int]$BatchSize = 8,
    [int]$ImgSize = 256,
    [float]$Lr = 0.01,
    [float]$FedProxMu = 0.01
)

$COMMON_ARGS = @(
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
    "--checkpoint_root", "results/fed_bcd",
    "--seed", "42"
)

# ─── Partition files ───
$PARTITIONS = @(
    @{name="source";       file="partitions/partition_source.json"},
    @{name="dirichlet_01"; file="partitions/partition_dirichlet_a0.1_n7.json"},
    @{name="dirichlet_05"; file="partitions/partition_dirichlet_a0.5_n7.json"},
    @{name="dirichlet_10"; file="partitions/partition_dirichlet_a1.0_n7.json"},
    @{name="iid";          file="partitions/partition_dirichlet_a100.0_n7.json"},
    @{name="hybrid";       file="partitions/partition_hybrid_c1_separate.json"}
)

# ─── FedAvg experiments (fedprox_mu=0) ───
Write-Host "`n========== FedAvg Experiments ==========" -ForegroundColor Cyan
foreach ($p in $PARTITIONS) {
    $projName = "FedAvg_$($p.name)_bcd"
    Write-Host "`n>>> Running $projName ..." -ForegroundColor Yellow
    $args = @("--partition_json", $p.file, "--project_name", $projName, "--fedprox_mu", "0.0") + $COMMON_ARGS
    python -m fed_cd.federated.fed_main @args
}

# ─── FedProx experiments (fedprox_mu=0.01) ───
Write-Host "`n========== FedProx Experiments (mu=$FedProxMu) ==========" -ForegroundColor Cyan
foreach ($p in $PARTITIONS) {
    $projName = "FedProx${FedProxMu}_$($p.name)_bcd"
    Write-Host "`n>>> Running $projName ..." -ForegroundColor Yellow
    $args = @("--partition_json", $p.file, "--project_name", $projName, "--fedprox_mu", $FedProxMu) + $COMMON_ARGS
    python -m fed_cd.federated.fed_main @args
}

Write-Host "`n========== All BCD experiments complete ==========" -ForegroundColor Green
