"""Tests for the report renderer.

Covers: normal render (mvnmle bootstrap), table formatting edge cases, and the
fail-loud behavior when a manifest is missing or contradictory (Bible rules 1 & 7).
Runnable in isolation: `pytest tests/test_render.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import render_report
from render import tables
from render.load import load

REPO = Path(__file__).resolve().parent.parent


# --- normal case ---------------------------------------------------------------

def test_mvnmle_renders_and_pins_version():
    md = render_report.build_for_test("mvnmle", "3.18.0")
    assert "pystatistics `mvnmle` v3.18.0" in md
    # All seven question headers present.
    for q in ["## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."]:
        assert q in md, f"missing section {q}"
    # A real number from the frozen threeway artifact made it into the report.
    assert "cpu_cholesky_fp64" in md and "gpu_mps_fp32" in md
    # PyPI provenance is asserted in the report body.
    assert "installed from **pypi**" in md


def test_manifest_matches_request():
    loaded = load("mvnmle", "3.18.0")
    assert loaded.manifest["subsystem"] == "mvnmle"
    assert loaded.manifest["pystatistics_version"] == "3.18.0"
    assert loaded.manifest["install_source"] == "pypi"


# --- edge cases ----------------------------------------------------------------

def test_md_table_empty_rows():
    assert "no rows" in tables.md_table(["a", "b"], [])


def test_fmt_handles_none_bool_float():
    assert tables._fmt(None) == ""
    assert tables._fmt(True) == "yes"
    assert tables._fmt(1234.5) == "1234" or tables._fmt(1234.5) == "1234.5" \
        or tables._fmt(1234.5).startswith("1234")
    assert tables._fmt("r_mvnmle") == "r_mvnmle"


def test_device_pivot_pairs_cpu_and_acc():
    recs = [
        {"engine": "pystatistics:cpu", "survey": "x", "p": 5, "n": 10,
         "loglik": -100.0, "wall_median_s": 1.0},
        {"engine": "pystatistics:gpu", "survey": "x", "p": 5, "n": 10,
         "loglik": -100.0, "wall_median_s": 0.5},
    ]
    out = tables.device_pivot(recs)
    assert "2.0x" in out  # speedup 1.0/0.5


def test_device_pivot_without_loglik_uses_dataset_key():
    recs = [
        {"engine": "pystatistics:cpu", "dataset": "n2000", "p": 20, "n": 2000,
         "wall_median_s": 4.0, "peak_mem_mb": 100.0},
        {"engine": "pystatistics:gpu", "dataset": "n2000", "p": 20, "n": 2000,
         "wall_median_s": 1.0, "peak_mem_mb": 250.0},
    ]
    out = tables.device_pivot(recs)
    assert "4.0x" in out and "loglik" not in out and "mem cpu (MB)" in out


# --- failure cases -------------------------------------------------------------

def test_missing_subsystem_raises():
    with pytest.raises(FileNotFoundError):
        load("does_not_exist", "9.9.9")


def test_version_mismatch_raises(tmp_path, monkeypatch):
    # Point ARTIFACTS at a temp tree with a deliberately wrong version inside.
    import render.load as L
    sub = tmp_path / "foo" / "v1.0.0"
    sub.mkdir(parents=True)
    (sub / "manifest.json").write_text(json.dumps(
        {"schema": "validation-artifact-manifest/v1", "subsystem": "foo",
         "pystatistics_version": "2.0.0", "install_source": "pypi", "studies": []}))
    (tmp_path / "subsystems" / "foo").mkdir(parents=True)
    (tmp_path / "subsystems" / "foo" / "meta.json").write_text("{}")
    monkeypatch.setattr(L, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(L, "SUBSYSTEMS", tmp_path / "subsystems")
    with pytest.raises(ValueError, match="version"):
        L.load("foo", "1.0.0")
