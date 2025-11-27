# Usage: .\scripts\nnunet_env.ps1
# Sets nnU-Net v2 directories to the local data/nnunet layout for the current PowerShell session.

$root = (Resolve-Path "$PSScriptRoot/..").Path
$env:nnUNet_raw = (Join-Path $root "data/nnunet/nnUNet_raw")
$env:nnUNet_preprocessed = (Join-Path $root "data/nnunet/nnUNet_preprocessed")
$env:nnUNet_results = (Join-Path $root "data/nnunet/nnUNet_results")

Write-Output "nnUNet_raw          = $($env:nnUNet_raw)"
Write-Output "nnUNet_preprocessed = $($env:nnUNet_preprocessed)"
Write-Output "nnUNet_results      = $($env:nnUNet_results)"

