# Setup environment for Offroad Segmentation
# This script creates a conda environment and installs dependencies

$env_name = "EDU"

Write-Host "Creating conda environment: $env_name" -ForegroundColor Cyan
conda create -n $env_name python=3.10 -y

Write-Host "Activating environment..." -ForegroundColor Cyan
# Conda activation in PowerShell can be tricky, we'll use the conda shell function if available
conda activate $env_name

Write-Host "Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Write-Host "Installing project requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "To activate this environment, run: conda activate EDU"
