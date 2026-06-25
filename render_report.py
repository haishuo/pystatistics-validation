#!/usr/bin/env python3
"""Render a version-pinned validation report from frozen artifacts.

Usage:
    python render_report.py <subsystem> <version> [--out reports/<sub>-v<ver>.md]
    python render_report.py --all          # render every (subsystem, version) found

The report is the human-readable RENDERING of a subsystem's frozen evidence at one
library version. Numbers come only from artifacts/; prose from subsystems/.
This script is the GENERATION path — reports are never hand-written.

# NON-DETERMINISTIC: the rendered-timestamp line uses the wall clock. Pass
# --stamp to fix it (used by tests so output is byte-stable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from render.load import ARTIFACTS, load
from render.report import build


def _default_out(subsystem: str, version: str) -> Path:
    return Path(__file__).resolve().parent / "reports" / f"{subsystem}-v{version}.md"


def _discover() -> list[tuple[str, str]]:
    found = []
    for sub_dir in sorted(ARTIFACTS.glob("*")):
        if not sub_dir.is_dir() or sub_dir.name == "schema":
            continue
        for ver_dir in sorted(sub_dir.glob("v*")):
            if (ver_dir / "manifest.json").is_file():
                found.append((sub_dir.name, ver_dir.name.lstrip("v")))
    return found


def build_for_test(subsystem: str, version: str, stamp: str = "TEST") -> str:
    """Render to a string without writing to disk (used by the test suite)."""
    return build(load(subsystem, version), rendered_utc=stamp)


def render_one(subsystem: str, version: str, out: Path | None, stamp: str) -> Path:
    loaded = load(subsystem, version)
    md = build(loaded, rendered_utc=stamp)
    out = out or _default_out(subsystem, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("subsystem", nargs="?")
    p.add_argument("version", nargs="?")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--all", action="store_true", help="render every artifact set found")
    p.add_argument("--stamp", default=None,
                   help="fixed render timestamp (default: now, UTC)")
    args = p.parse_args(argv)

    if args.stamp is None:
        # NON-DETERMINISTIC: wall-clock stamp for human-facing provenance only.
        from datetime import datetime, timezone
        args.stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if args.all:
        targets = _discover()
        if not targets:
            print("no artifact sets found under artifacts/", file=sys.stderr)
            return 1
    else:
        if not (args.subsystem and args.version):
            p.error("give <subsystem> <version>, or --all")
        targets = [(args.subsystem, args.version)]

    for subsystem, version in targets:
        out = render_one(subsystem, version, args.out if not args.all else None, args.stamp)
        print(f"rendered {subsystem} v{version} -> {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
