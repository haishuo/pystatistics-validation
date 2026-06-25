"""Load frozen artifacts and subsystem prose for one (subsystem, version).

One job: file I/O + structural validation for the renderer. Fails loud (raises) on
a missing/contradictory manifest rather than rendering a silently-incomplete report.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
SUBSYSTEMS = REPO_ROOT / "subsystems"


@dataclass(frozen=True)
class Loaded:
    subsystem: str
    version: str
    version_dir: Path
    manifest: dict[str, Any]
    meta: dict[str, Any]


def load(subsystem: str, version: str) -> Loaded:
    """Load manifest + subsystem meta, validating the cross-references.

    Raises
    ------
    FileNotFoundError
        If the version directory, manifest, or subsystem meta is missing.
    ValueError
        If the manifest's subsystem/version disagree with the request.
    """
    version_dir = ARTIFACTS / subsystem / f"v{version}"
    manifest_path = version_dir / "manifest.json"
    meta_path = SUBSYSTEMS / subsystem / "meta.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"no subsystem meta at {meta_path}")

    manifest = json.loads(manifest_path.read_text())
    meta = json.loads(meta_path.read_text())

    if manifest.get("subsystem") != subsystem:
        raise ValueError(
            f"manifest subsystem {manifest.get('subsystem')!r} != requested {subsystem!r}")
    if manifest.get("pystatistics_version") != version:
        raise ValueError(
            f"manifest version {manifest.get('pystatistics_version')!r} != requested {version!r}")
    return Loaded(subsystem, version, version_dir, manifest, meta)


def read_records(version_dir: Path, rel: str) -> list[dict[str, Any]]:
    """Read the `records` array from a run JSON (or [] if absent)."""
    path = version_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"run file referenced by manifest not found: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("records", [])


def read_csv(version_dir: Path, rel: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read a summary CSV → (header, rows-as-dicts)."""
    path = version_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"summary file referenced by manifest not found: {path}")
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return list(header), rows
