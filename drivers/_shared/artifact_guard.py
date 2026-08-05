"""Refuse to silently overwrite blessed validation evidence.

Every driver hardcodes its artifact directory as
``<repo>/artifacts/<module>/v<version>/runs`` with no environment override, so
running any driver writes straight on top of whatever evidence is already there.
With a released version installed, that means a casual "let me just check
something" run destroys the artifacts a report was blessed against — silently,
with no prompt and no warning.

The guard keys on **git-tracked**, not merely "file exists". Overwriting a fresh
run you produced five minutes ago is ordinary work; overwriting evidence that is
committed to the repository is the destructive act worth stopping. A tracked
file is the repo's record of a validated run, so that is the line.

Set ``PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1`` to proceed anyway — which is what
a genuine re-validation does. The point is not to forbid regeneration; it is to
make regeneration deliberate and visible rather than accidental.

Usage — drop-in for ``pystatsval.serialize.write_run``::

    from artifact_guard import write_run          # instead of pystatsval.serialize

For writers that do not go through ``write_run``::

    from artifact_guard import guard_artifact_path
    guard_artifact_path(out)
    out.write_text(...)
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Set to "1" to permit overwriting committed evidence (a real re-validation).
ALLOW_OVERWRITE_ENV = "PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE"


def _overwrite_allowed() -> bool:
    return os.environ.get(ALLOW_OVERWRITE_ENV, "").strip() not in ("", "0")


@lru_cache(maxsize=None)
def _repo_root(start: Path) -> Path | None:
    """Return the git repo root containing ``start``, or ``None`` if not a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out.stdout.strip()) if out.returncode == 0 else None


@lru_cache(maxsize=None)
def _tracked_files(repo: Path) -> frozenset[str]:
    """Return every git-tracked path in ``repo``, repo-relative.

    Read once per process and cached: a driver may write a hundred artifacts and
    should not pay a subprocess per write.
    """
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    if out.returncode != 0:
        return frozenset()
    return frozenset(out.stdout.splitlines())


def is_tracked(path: str | Path) -> bool:
    """True if ``path`` is committed to git (i.e. is part of the repo's record)."""
    p = Path(path).resolve()
    repo = _repo_root(p.parent if p.parent.exists() else Path.cwd())
    if repo is None:
        return False
    try:
        rel = p.relative_to(repo.resolve())
    except ValueError:
        return False
    return str(rel) in _tracked_files(repo)


def guard_artifact_path(path: str | Path) -> Path:
    """Fail loud if ``path`` is committed evidence and overwrite is not allowed.

    Raises
    ------
    RuntimeError
        If the target is git-tracked and ``PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE``
        is unset. Untracked targets, and any target when the variable is set,
        pass through untouched.
    """
    p = Path(path)
    if _overwrite_allowed() or not is_tracked(p):
        return p
    raise RuntimeError(
        f"refusing to overwrite committed validation evidence: {p}\n"
        f"This file is tracked in git, so it is the record of a run something "
        f"was blessed against. If this IS a deliberate re-validation, set "
        f"{ALLOW_OVERWRITE_ENV}=1. If you are only checking something, run in a "
        f"scratch git worktree so the real artifacts are untouched.")


def write_run(path: str | Path, run: dict[str, Any]) -> Path:
    """Guarded drop-in for ``pystatsval.serialize.write_run``.

    Applies :func:`guard_artifact_path`, then delegates — so schema validation
    and the actual write remain owned by ``pystatsval``, not reimplemented here.
    """
    from pystatsval.serialize import write_run as _write_run

    guard_artifact_path(path)
    return _write_run(path, run)
