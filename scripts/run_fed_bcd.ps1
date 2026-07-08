# Federated BCD experiments: FedAvg and FedProx on multiple Non-IID partitions,
# repeated over multiple random seeds for mean±std reporting.
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_fed_bcd.ps1

param(
    [int]$Epochs = 200,
    [int]$FracNum = 5,
    [int]$LocalEp = 2,
    [int]$BatchSize = 8,
    [int]$ImgSize = 256,
    [float]$Lr = 0.01,
    [float]$FedProxMu = 0.01,
    [int[]]$Seeds = @(42, 2024, 0)
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

foreach ($seed in $Seeds) {
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
        "--seed", "$seed"
    )

    Write-Host "`n========== Seed = $seed ==========" -ForegroundColor Cyan

    # ─── FedAvg experiments (fed_alg=fedavg) ───
    Write-Host "--- FedAvg (seed=$seed) ---"
    foreach ($p in $PARTITIONS) {
        $projName = "FedAvg_$($p.name)_bcd_s$seed"
        Write-Host ">>> $projName" -ForegroundColor Yellow
        $args = @("--partition_json", $p.file, "--project_name", $projName,
                  "--fed_alg", "fedavg", "--fedprox_mu", "0.0", "--iid", "False") + $COMMON_ARGS
        python -m fed_cd.federated.fed_main @args
    }

    # ─── FedProx experiments (fed_alg=fedprox) ───
    Write-Host "--- FedProx mu=$FedProxMu (seed=$seed) ---"
    foreach ($p in $PARTITIONS) {
        $projName = "FedProx${FedProxMu}_$($p.name)_bcd_s$seed"
        Write-Host ">>> $projName" -ForegroundColor Yellow
        $args = @("--partition_json", $p.file, "--project_name", $projName,
                  "--fed_alg", "fedprox", "--fedprox_mu", "$FedProxMu", "--iid", "False") + $COMMON_ARGS
        python -m fed_cd.federated.fed_main @args
    }
}

Write-Host "`n========== All BCD experiments complete ==========" -ForegroundColor Green
Write-Host "汇总: python -m fed_cd.summarize --results_root results/fed_bcd"
