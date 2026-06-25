#!/bin/bash
set -e

echo "=== Creating conda environment 'test' ==="
conda create -n test python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate test

# Install from LOCAL source, not PyPI — we're testing the latest code.
# Resolve paths relative to this script's location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEV_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Installing pystatistics from local source ==="
pip install -e "${DEV_DIR}/pystatistics[gpu]"

echo "=== Installing pystatsbio from local source ==="
pip install -e "${DEV_DIR}/pystatsbio[gpu]"

echo "=== Installing test dependencies ==="
pip install pytest pytest-benchmark scikit-learn pandas pyreadstat

echo "=== Installing R packages ==="
Rscript -e 'install.packages(c(
    "jsonlite", "moments", "car", "survival", "lme4", "lmerTest", "boot",
    "pwr", "drc", "pROC", "PKNCA", "epiR"
), repos="https://cloud.r-project.org")'

echo "=== Setup complete ==="
echo "Activate with: conda activate test"
