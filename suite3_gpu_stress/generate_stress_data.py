#!/usr/bin/env python3
"""
Prepare TCGA BRCA data for GPU stress testing.

Loads the preprocessed TCGA npz and creates test-ready arrays:
- Full expression matrix for batch AUC (20K genes)
- Top 500 genes for bootstrap and GLM
- Labels (ER status)
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed TCGA data
    npz_path = DATA_DIR / "tcga_brca_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Run data/download_datasets.py first to create {npz_path}"
        )

    data = np.load(npz_path, allow_pickle=True)
    expression = data["expression"]  # (n_samples, n_genes) float32
    labels = data["labels"]          # (n_samples,) int32
    gene_names = data["gene_names"]  # (n_genes,) str

    n_samples, n_genes = expression.shape
    print(f"TCGA BRCA: {n_samples} samples, {n_genes} genes")
    print(f"Labels: {np.bincount(labels)} (0=neg, 1=pos)")

    # Save full matrix for batch AUC
    np.savez_compressed(
        FIXTURES_DIR / "tcga_full.npz",
        expression=expression,
        labels=labels,
        gene_names=gene_names,
    )
    print(f"Saved full matrix: {expression.shape}")

    # Compute per-gene AUC quickly to identify top genes
    # Simple rank-based AUC (Mann-Whitney U)
    from scipy.stats import mannwhitneyu

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    print(f"Computing quick AUC for {n_genes} genes to find top 500...")
    aucs = np.zeros(n_genes)
    for j in range(n_genes):
        try:
            u, _ = mannwhitneyu(
                expression[pos_mask, j], expression[neg_mask, j],
                alternative="two-sided",
            )
            aucs[j] = u / (n_pos * n_neg)
        except Exception:
            aucs[j] = 0.5

    # Convert to "directional AUC" (max of AUC, 1-AUC)
    aucs_dir = np.maximum(aucs, 1 - aucs)

    # Top 500 by discriminative power
    top500_idx = np.argsort(aucs_dir)[-500:]
    top500_expr = expression[:, top500_idx].astype(np.float64)
    top500_genes = gene_names[top500_idx]
    top500_aucs = aucs_dir[top500_idx]

    np.savez_compressed(
        FIXTURES_DIR / "tcga_top500.npz",
        expression=top500_expr,
        labels=labels,
        gene_names=top500_genes,
        aucs=top500_aucs,
        top500_idx=top500_idx,
    )
    print(f"Saved top 500 genes: {top500_expr.shape}")
    print(f"Top 500 AUC range: {top500_aucs.min():.4f} - {top500_aucs.max():.4f}")

    # Also save as CSV for R
    import pandas as pd

    # Full expression (too large for CSV — save top 500 only)
    top500_df = pd.DataFrame(top500_expr, columns=[str(g) for g in top500_genes])
    top500_df["label"] = labels
    top500_df.to_csv(FIXTURES_DIR / "tcga_top500.csv", index=False)
    print(f"Saved top 500 CSV for R: {top500_df.shape}")

    # Full expression as CSV (for R batch AUC — this will be large)
    # Only save a random subset of genes if too many
    if n_genes > 20000:
        rng = np.random.default_rng(42)
        gene_subset = rng.choice(n_genes, 20000, replace=False)
    else:
        gene_subset = np.arange(n_genes)

    full_df = pd.DataFrame(
        expression[:, gene_subset],
        columns=[str(gene_names[g]) for g in gene_subset],
    )
    full_df["label"] = labels
    full_df.to_csv(FIXTURES_DIR / "tcga_full_for_r.csv", index=False)
    print(f"Saved full CSV for R: {full_df.shape}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
