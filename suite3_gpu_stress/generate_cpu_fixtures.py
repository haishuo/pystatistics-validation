#!/usr/bin/env python3
"""
Generate CPU fixtures for GPU stress comparison.

Runs the same workloads as the GPU tests but on CPU, saving results
and wall-clock timings for speedup comparison.

Run from /mnt/projects/test:
    conda activate test
    python suite3_gpu_stress/generate_cpu_fixtures.py
"""

import json
import time
import numpy as np
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def run_batch_auc_cpu():
    """Compute batch AUC on CPU for all genes."""
    from pystatsbio.diagnostic import batch_auc

    data = np.load(FIXTURES_DIR / "tcga_full.npz", allow_pickle=True)
    expression = data["expression"].astype(np.float64)
    labels = data["labels"]

    print(f"  Batch AUC CPU: {expression.shape[1]} genes x {expression.shape[0]} samples")

    t_start = time.perf_counter()
    result = batch_auc(labels, expression, backend="cpu")
    elapsed = time.perf_counter() - t_start

    print(f"  Done: {elapsed:.1f} seconds")
    return {
        "auc": result.auc.tolist(),
        "se": result.se.tolist(),
        "n_markers": result.n_markers,
    }, elapsed


def run_glm_cpu():
    """Fit GLM binomial on CPU with top 500 genes."""
    from pystatistics import DataSource
    from pystatistics.regression import Design, fit

    data = np.load(FIXTURES_DIR / "tcga_top500.npz", allow_pickle=True)
    expression = data["expression"]  # float64
    labels = data["labels"]

    # Add intercept
    X = np.column_stack([np.ones(len(labels)), expression])
    y = labels.astype(np.float64)

    print(f"  GLM CPU: {X.shape[0]} samples x {X.shape[1]} predictors")

    design = Design.from_arrays(X, y)

    t_start = time.perf_counter()
    result = fit(design, family="binomial", backend="cpu")
    elapsed = time.perf_counter() - t_start

    print(f"  Done: {elapsed:.1f} seconds")
    return {
        "coefficients": result.coefficients.tolist(),
        "deviance": float(result.deviance),
    }, elapsed


def run_bootstrap_cpu():
    """Bootstrap AUC for top gene on CPU."""
    from pystatistics.montecarlo import boot

    data = np.load(FIXTURES_DIR / "tcga_full.npz", allow_pickle=True)
    expression = data["expression"]
    labels = data["labels"]

    # Use the gene with highest variance as proxy for "top gene"
    gene_var = np.var(expression, axis=0)
    top_gene_idx = np.argmax(gene_var)
    top_gene_vals = expression[:, top_gene_idx].astype(np.float64)

    # Stack labels and gene values for boot
    boot_data = np.column_stack([labels, top_gene_vals])

    def auc_stat(data, indices):
        d = data[indices]
        y = d[:, 0]
        x = d[:, 1]
        n1 = np.sum(y == 1)
        n0 = np.sum(y == 0)
        if n1 == 0 or n0 == 0:
            return np.array([0.5])
        # Mann-Whitney U statistic
        ranks = np.argsort(np.argsort(x)) + 1.0
        u = np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2
        return np.array([u / (n1 * n0)])

    print(f"  Bootstrap CPU: 5000 resamples, {len(labels)} observations")

    t_start = time.perf_counter()
    result = boot(boot_data, auc_stat, R=5000, seed=42, backend="cpu")
    elapsed = time.perf_counter() - t_start

    print(f"  Done: {elapsed:.1f} seconds")
    return {
        "t0": result.t0.tolist(),
        "se": result.se.tolist(),
    }, elapsed


def run_permutation_cpu():
    """Permutation test for top gene on CPU."""
    from pystatistics.montecarlo import permutation_test

    data = np.load(FIXTURES_DIR / "tcga_full.npz", allow_pickle=True)
    expression = data["expression"]
    labels = data["labels"]

    gene_var = np.var(expression, axis=0)
    top_gene_idx = np.argmax(gene_var)

    pos = expression[labels == 1, top_gene_idx].astype(np.float64)
    neg = expression[labels == 0, top_gene_idx].astype(np.float64)

    def diff_means(x, y):
        return np.mean(x) - np.mean(y)

    print(f"  Permutation CPU: 50000 permutations")

    t_start = time.perf_counter()
    result = permutation_test(pos, neg, diff_means, R=50000, seed=43, backend="cpu")
    elapsed = time.perf_counter() - t_start

    print(f"  Done: {elapsed:.1f} seconds")
    return {
        "observed_stat": float(result.observed_stat),
        "p_value": float(result.p_value),
    }, elapsed


def main():
    print("=== Generating CPU fixtures for GPU comparison ===\n")

    cpu_results = {}
    cpu_timings = {}

    workloads = [
        ("batch_auc", run_batch_auc_cpu),
        ("glm", run_glm_cpu),
        ("bootstrap", run_bootstrap_cpu),
        ("permutation", run_permutation_cpu),
    ]

    for name, fn in workloads:
        print(f"\n--- {name} ---")
        try:
            result, elapsed = fn()
            cpu_results[name] = result
            cpu_timings[name] = elapsed
        except Exception as e:
            print(f"  FAILED: {e}")
            cpu_timings[name] = None

    # Save
    with open(FIXTURES_DIR / "cpu_results.json", "w") as f:
        json.dump(cpu_results, f, indent=2)

    with open(FIXTURES_DIR / "cpu_timings.json", "w") as f:
        json.dump(cpu_timings, f, indent=2)

    print("\n=== CPU Timings ===")
    for name, t in cpu_timings.items():
        if t is not None:
            print(f"  {name}: {t:.1f} seconds")
        else:
            print(f"  {name}: FAILED")

    print("\nDone.")


if __name__ == "__main__":
    main()
