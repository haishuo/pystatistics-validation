#!/usr/bin/env python3
"""
Prepare California Housing data for Suite 1 tests.

Derives additional columns needed for ANOVA, logistic regression,
chi-squared tests, survival proxy, and mixed models.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    raw = pd.read_csv(DATA_DIR / "california_housing.csv")
    print(f"Raw California Housing: {raw.shape}")

    df = raw.copy()

    # region: categorical from Latitude bins (North/Central/South)
    lat_33 = np.percentile(df["Latitude"], 33)
    lat_67 = np.percentile(df["Latitude"], 67)
    df["region"] = pd.cut(
        df["Latitude"],
        bins=[-np.inf, lat_33, lat_67, np.inf],
        labels=["South", "Central", "North"],
    ).astype(str)

    # old_house: binary from HouseAge > median
    df["old_house"] = (df["HouseAge"] > df["HouseAge"].median()).astype(int)

    # high_value: binary from MedHouseVal > median
    df["high_value"] = (df["MedHouseVal"] > df["MedHouseVal"].median()).astype(int)

    # pop_quartile: 4-level categorical from Population quartiles
    df["pop_quartile"] = pd.qcut(
        df["Population"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)

    # block_id: geographic grid blocks for mixed model random effect
    # Create ~200 blocks by binning Latitude and Longitude into 14x14 grid
    lat_bins = pd.cut(df["Latitude"], bins=14, labels=False)
    lon_bins = pd.cut(df["Longitude"], bins=14, labels=False)
    df["block_id"] = (lat_bins * 14 + lon_bins).astype(str)

    # Save prepared data
    out_path = FIXTURES_DIR / "california_prepared.csv"
    df.to_csv(out_path, index=False)
    print(f"Prepared data: {df.shape} -> {out_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Regions: {df['region'].value_counts().to_dict()}")
    print(f"  high_value: {df['high_value'].value_counts().to_dict()}")
    print(f"  Blocks: {df['block_id'].nunique()}")


if __name__ == "__main__":
    main()
