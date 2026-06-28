# Centralized BCD experiment: performance upper bound for the federated runs.
# Trains a single BIT-CD model on the union of all training samples
# (gcd + ugcd_full + ucd + ugd) with the SAME hyperparameters as run_fed_bcd.ps1
# — the only difference is centralized vs. federated optimization.
#
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_centralized.ps1

param(
    [int]$Epochs = 200,
    [int]$BatchSize = 8,
    [int]$ImgSize = 256,
    [float]$Lr = 0.01,
    [string]$NetG = "base_transformer_pos_s4_dd8",
    [string]$Tag = "base"
)

$ProjName = "Centr_${Tag}_bcd"

Write-Host "`n========== Centralized BCD Experiment ==========" -ForegroundColor Cyan
Write-Host ">>> Model: $NetG" -ForegroundColor Yellow
Write-Host ">>> Project: $ProjName" -ForegroundColor Yellow
Write-Host ""

$args = @(
    "--data_root", "../WHU-GCD",
    "--net_G", $NetG,
    "--num_classes", "2",
    "--img_size", $ImgSize,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--lr", $Lr,
    "--lr_policy", "linear",
    "--optimizer", "sgd",
    "--pretrained", "True",
    "--eval_splits", "val,test,test2",
    "--global_test_frequency", "20",
    "--save_frequency", "20",
    "--checkpoint_root", "results/centralized",
    "--project_name", $ProjName,
    "--seed", "42"
)
python -m fed_cd.centralized.cen_main @args

Write-Host "`n========== Centralized BCD experiment complete ==========" -ForegroundColor Green
