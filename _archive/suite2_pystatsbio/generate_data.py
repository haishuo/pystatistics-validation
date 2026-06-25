#!/usr/bin/env python3
"""
Prepare data for Suite 2 (PyStatsBio) tests.

- NHANES is already downloaded as CSV
- Generate synthetic dose-response data (50 compounds, 8 doses, LL.4)
- Generate synthetic PK profiles (5 subjects, oral one-compartment)
"""

import numpy as np
import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RNG = np.random.default_rng(20250415)


def generate_dose_response():
    """Generate synthetic HTS dose-response data: 50 compounds x 8 doses."""
    print("\n=== Generating dose-response data ===")

    K = 50  # compounds
    N = 8   # doses per compound
    dose = np.array([0, 0.001, 0.01, 0.1, 1, 10, 100, 1000], dtype=float)

    # Randomize parameters per compound
    ec50_true = np.logspace(-2, 2, K)  # 0.01 to 100
    hill_true = RNG.uniform(0.8, 2.5, K)
    bottom_true = RNG.uniform(5, 15, K)
    top_true = RNG.uniform(80, 100, K)

    dose_matrix = np.tile(dose, (K, 1))
    response_matrix = np.zeros((K, N))

    for i in range(K):
        # LL.4: response = bottom + (top - bottom) / (1 + (dose/ec50)^hill)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dose > 0, (dose / ec50_true[i]) ** hill_true[i], 0.0)
        response_matrix[i] = (
            bottom_true[i]
            + (top_true[i] - bottom_true[i]) / (1 + ratio)
            + RNG.normal(0, 2, N)
        )

    # Save
    np.savez(
        FIXTURES_DIR / "dose_response_data.npz",
        dose_matrix=dose_matrix,
        response_matrix=response_matrix,
        ec50_true=ec50_true,
        hill_true=hill_true,
        bottom_true=bottom_true,
        top_true=top_true,
    )

    # Also save individual compounds as JSON for R
    for i in [0, 10, 20, 30, 40]:  # 5 representative compounds
        compound = {
            "name": f"compound_{i:03d}",
            "dose": dose.tolist(),
            "response": response_matrix[i].tolist(),
            "true_ec50": float(ec50_true[i]),
            "true_hill": float(hill_true[i]),
            "true_bottom": float(bottom_true[i]),
            "true_top": float(top_true[i]),
        }
        with open(FIXTURES_DIR / f"compound_{i:03d}.json", "w") as f:
            json.dump(compound, f, indent=2)

    print(f"  {K} compounds, {N} doses each")
    print(f"  EC50 range: {ec50_true.min():.4f} to {ec50_true.max():.4f}")


def generate_pk_profiles():
    """Generate synthetic PK profiles: oral one-compartment model."""
    print("\n=== Generating PK profiles ===")

    # 5 subjects with slightly different PK parameters
    subjects = []
    for i in range(5):
        ka = 1.0 + RNG.uniform(-0.2, 0.2)   # absorption rate
        ke = 0.1 + RNG.uniform(-0.02, 0.02)  # elimination rate
        V = 10.0 + RNG.uniform(-2, 2)        # volume of distribution
        F = 1.0                               # bioavailability
        D = 100.0                             # dose

        time = np.array([0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24], dtype=float)
        A = F * D * ka / (V * (ka - ke))
        conc = A * (np.exp(-ke * time) - np.exp(-ka * time))
        conc = np.maximum(conc, 0)  # no negative concentrations

        subjects.append({
            "time": time.tolist(),
            "concentration": conc.tolist(),
            "dose": D,
            "route": "ev",
            "true_ka": ka,
            "true_ke": ke,
            "true_V": V,
        })

    # Also add one IV bolus subject
    ke_iv = 0.1
    V_iv = 10.0
    D_iv = 100.0
    time_iv = np.array([0, 0.083, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24], dtype=float)
    conc_iv = (D_iv / V_iv) * np.exp(-ke_iv * time_iv)

    subjects.append({
        "time": time_iv.tolist(),
        "concentration": conc_iv.tolist(),
        "dose": D_iv,
        "route": "iv",
        "true_ke": ke_iv,
        "true_V": V_iv,
    })

    np.savez(
        FIXTURES_DIR / "pk_data.npz",
        subjects=np.array(subjects, dtype=object),
    )

    # Save as JSON for R
    with open(FIXTURES_DIR / "pk_subjects.json", "w") as f:
        json.dump(subjects, f, indent=2)

    print(f"  {len(subjects)} subjects ({len(subjects)-1} oral, 1 IV)")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_dose_response()
    generate_pk_profiles()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
