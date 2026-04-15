#!/usr/bin/env python3
"""
Download all datasets needed for the three test suites.

Suite 1: California Housing (sklearn/StatLib) — general statistics
Suite 2: NHANES 2017-2018 (CDC) — biostatistics
Suite 3: TCGA BRCA RNA-seq (UCSC Xena/GDC) — GPU stress test
"""

import os
import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
SUITE1_FIXTURES = DATA_DIR.parent / "suite1_pystatistics" / "fixtures"
SUITE2_FIXTURES = DATA_DIR.parent / "suite2_pystatsbio" / "fixtures"
SUITE3_FIXTURES = DATA_DIR.parent / "suite3_gpu_stress" / "fixtures"


def download_file(url, dest, description=""):
    """Download a file with progress indication."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [cached] {dest.name}")
        return
    print(f"  Downloading {description or dest.name}...")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; pystatistics-test/1.0)",
        })
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────
# Suite 1: California Housing
# ──────────────────────────────────────────────────────────────────────

def download_california_housing():
    """Download California Housing via sklearn."""
    print("\n=== Suite 1: California Housing ===")
    dest = DATA_DIR / "california_housing.csv"
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return

    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv(dest, index=False)
    print(f"  -> {dest} ({len(df)} rows, {len(df.columns)} columns)")


# ──────────────────────────────────────────────────────────────────────
# Suite 2: NHANES 2017-2018
# ──────────────────────────────────────────────────────────────────────

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

NHANES_FILES = {
    "DEMO_J.XPT": f"{NHANES_BASE}/DEMO_J.XPT",
    "GHB_J.XPT": f"{NHANES_BASE}/GHB_J.XPT",
    "BIOPRO_J.XPT": f"{NHANES_BASE}/BIOPRO_J.XPT",
}


def download_nhanes():
    """Download NHANES 2017-2018 XPT files and merge into CSV."""
    print("\n=== Suite 2: NHANES 2017-2018 ===")
    merged_dest = DATA_DIR / "nhanes_biomarker.csv"
    if merged_dest.exists():
        print(f"  [cached] {merged_dest.name}")
        return

    import pyreadstat

    dfs = {}
    for fname, url in NHANES_FILES.items():
        xpt_path = DATA_DIR / fname
        download_file(url, xpt_path, fname)
        df, meta = pyreadstat.read_xport(str(xpt_path))
        dfs[fname] = df
        print(f"    {fname}: {len(df)} rows, {len(df.columns)} columns")

    # Merge on SEQN
    merged = dfs["DEMO_J.XPT"]
    for fname in ["GHB_J.XPT", "BIOPRO_J.XPT"]:
        merged = merged.merge(dfs[fname], on="SEQN", how="inner")

    # Keep useful columns
    keep_cols = [
        "SEQN", "RIDAGEYR", "RIAGENDR",  # demographics
        "LBXGH",  # HbA1c (glycohemoglobin)
        "LBXSATSI", "LBXSAL", "LBXSAPSI", "LBXSC3SI",  # albumin, ALP, creatinine
        "LBXSBU", "LBXSCA", "LBXSCH", "LBXSGL",  # BUN, calcium, cholesterol, glucose
        "LBXSIR", "LBXSKSI", "LBXSNASI", "LBXSPH",  # iron, potassium, sodium, phosphorus
        "LBXSTB", "LBXSTP", "LBXSTR", "LBXSUA",  # total bilirubin, protein, triglycerides, uric acid
    ]
    available = [c for c in keep_cols if c in merged.columns]
    merged = merged[available].dropna()

    # Derive diabetic label
    if "LBXGH" in merged.columns:
        merged["diabetic"] = (merged["LBXGH"] >= 6.5).astype(int)

    merged.to_csv(merged_dest, index=False)
    print(f"  -> {merged_dest} ({len(merged)} rows, {len(merged.columns)} columns)")


# ──────────────────────────────────────────────────────────────────────
# Suite 3: TCGA BRCA RNA-seq (UCSC Xena / GDC Hub)
# ──────────────────────────────────────────────────────────────────────

# UCSC Xena TCGA hub URLs (legacy hub — publicly accessible)
TCGA_EXPR_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.BRCA.sampleMap/HiSeqV2.gz"
)
TCGA_PHENO_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"
)


def download_tcga_brca():
    """Download TCGA BRCA gene expression and phenotype data."""
    print("\n=== Suite 3: TCGA BRCA RNA-seq ===")
    processed_dest = DATA_DIR / "tcga_brca_processed.npz"
    if processed_dest.exists():
        print(f"  [cached] {processed_dest.name}")
        return

    # Download files
    expr_gz = DATA_DIR / "TCGA-BRCA.HiSeqV2.gz"
    pheno_file = DATA_DIR / "TCGA-BRCA.clinicalMatrix"

    download_file(TCGA_EXPR_URL, expr_gz, "gene expression matrix (~64MB)")
    download_file(TCGA_PHENO_URL, pheno_file, "clinical matrix")

    # Load gene expression (HiSeqV2: genes as rows, samples as columns, gzipped)
    print("  Loading gene expression matrix...")
    expr_df = pd.read_csv(expr_gz, sep="\t", index_col=0, compression="gzip")
    print(f"    Raw: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")

    # Load phenotype (plain TSV)
    print("  Loading phenotype data...")
    pheno_df = pd.read_csv(pheno_file, sep="\t", index_col=0)

    # Find ER status column
    er_col = None
    for col in pheno_df.columns:
        if "estrogen" in col.lower() and "receptor" in col.lower():
            er_col = col
            break
    if er_col is None:
        # Try alternative column names
        for col in pheno_df.columns:
            if "er_status" in col.lower() or "er status" in col.lower():
                er_col = col
                break

    if er_col is None:
        print("  WARNING: Could not find ER status column. Using random binary label.")
        print(f"  Available columns: {list(pheno_df.columns[:20])}")
        # Fallback: use tumor/normal as label
        for col in pheno_df.columns:
            if "sample_type" in col.lower():
                er_col = col
                break

    # Match samples between expression and phenotype
    common_samples = list(set(expr_df.columns) & set(pheno_df.index))
    print(f"    Common samples: {len(common_samples)}")

    if er_col and len(common_samples) > 100:
        pheno_sub = pheno_df.loc[common_samples, er_col].dropna()

        # Create binary label
        unique_vals = pheno_sub.unique()
        print(f"    Label column '{er_col}': {unique_vals[:10]}")

        if any("positive" in str(v).lower() for v in unique_vals):
            binary = (pheno_sub.str.lower().str.contains("positive")).astype(int)
        elif any("primary" in str(v).lower() for v in unique_vals):
            binary = (pheno_sub.str.lower().str.contains("primary")).astype(int)
        else:
            # Generic: first unique value = 0, rest = 1
            binary = (pheno_sub != unique_vals[0]).astype(int)

        valid_samples = binary.dropna().index.tolist()
    else:
        # Fallback: use all samples, generate synthetic label
        valid_samples = common_samples[:min(len(common_samples), 1000)]
        rng = np.random.default_rng(42)
        binary = pd.Series(
            rng.binomial(1, 0.3, len(valid_samples)),
            index=valid_samples
        )

    # Subset expression to valid samples
    expr_sub = expr_df[valid_samples].T  # (n_samples, n_genes)
    labels = binary.loc[valid_samples].values.astype(np.int32)

    print(f"    Expression matrix: {expr_sub.shape}")
    print(f"    Label distribution: {np.bincount(labels)}")

    # Log-transform (log2(FPKM + 1))
    expr_vals = np.log2(expr_sub.values.astype(np.float64) + 1)

    # Filter low-variance genes (keep top 20000 by variance)
    gene_var = np.var(expr_vals, axis=0)
    n_keep = min(20000, expr_vals.shape[1])
    top_idx = np.argsort(gene_var)[-n_keep:]
    expr_filtered = expr_vals[:, top_idx]
    gene_names = np.array(expr_sub.columns)[top_idx]

    print(f"    After filtering: {expr_filtered.shape[0]} samples x {expr_filtered.shape[1]} genes")

    # Save as npz
    np.savez_compressed(
        processed_dest,
        expression=expr_filtered.astype(np.float32),
        labels=labels,
        gene_names=gene_names,
    )
    print(f"  -> {processed_dest} ({processed_dest.stat().st_size / 1e6:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_california_housing()
    download_nhanes()
    download_tcga_brca()
    print("\n=== All datasets downloaded ===")
