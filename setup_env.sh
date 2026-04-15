#!/bin/bash
set -e

echo "=== Creating conda environment 'test' ==="
conda create -n test python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate test

echo "=== Installing pystatistics and pystatsbio from PyPI ==="
pip install "pystatistics[gpu]" "pystatsbio[gpu]"

echo "=== Installing test dependencies ==="
pip install pytest pytest-benchmark scikit-learn pandas pyreadstat

echo "=== Installing R packages ==="
Rscript -e 'install.packages(c(
    "jsonlite", "moments", "car", "survival", "lme4", "lmerTest", "boot",
    "pwr", "drc", "pROC", "PKNCA", "epiR"
), repos="https://cloud.r-project.org")'

echo "=== Setup complete ==="
echo "Activate with: conda activate test"
