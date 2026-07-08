# Federated torchange baselines on WHU-GCD (binary change detection).
# Federates each torchange baseline with FedAvg and FedProx under the SAME Non-IID
# partition and schedule as the BIT-CD federated experiments, repeated over seeds.
#
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_fed_torchange_bcd.ps1

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

$MODELS = @(
    "changesparse_bcd",
    "changestar_1xd_r18",
    "changestar_1xd",
    "changestar_2_5"
)

# Canonical Non-IID partition for the main comparison table.
$PARTITIONS = @(
    @{name="dirichlet_05"; file="partitions/partition_dirichlet_a0.5_n7.json"}
    #@{name="dirichlet_01"; file="partitions/partition_dirichlet_a0.1_n7.json"}
    #@{name="source";       file="partitions/partition_source.json"}
    #@{name="iid";          file="partitions/partition_dirichlet_a100.0_n7.json"}
)

foreach ($seed in $Seeds) {
    Write-Host "`n========== Seed = $seed ==========" -ForegroundColor Cyan
    foreach ($netG in $MODELS) {
        foreach ($p in $PARTITIONS) {
            $common = @(
                "--partition_json", $p.file,
                "--data_root", "../WHU-GCD",
                "--net_G", $netG,
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
                "--iid", "False",
                "--eval_splits", "val,test,test2",
                "--global_test_frequency", "20",
                "--save_frequency", "20",
                "--checkpoint_root", "results/torchange_fed",
                "--seed", "$seed"
            )

            $projAvg = "FedAvg_${netG}_$($p.name)_bcd_s$seed"
            Write-Host ">>> $projAvg" -ForegroundColor Yellow
            $a = @("--project_name", $projAvg,
                   "--fed_alg", "fedavg", "--fedprox_mu", "0.0") + $common
            python -m fed_cd.federated.fed_main @a

            $projProx = "FedProx_${netG}_$($p.name)_bcd_s$seed"
            Write-Host ">>> $projProx" -ForegroundColor Yellow
            $a = @("--project_name", $projProx,
                   "--fed_alg", "fedprox", "--fedprox_mu", "$FedProxMu") + $common
            python -m fed_cd.federated.fed_main @a
        }
    }
}

Write-Host "`n========== All federated torchange baselines complete ==========" -ForegroundColor Green
Write-Host "汇总: python -m fed_cd.summarize --results_root results/torchange_fed"
