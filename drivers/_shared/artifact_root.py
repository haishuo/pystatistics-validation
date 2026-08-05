"""Decide where generated evidence is written.

One job: return the root directory for artifacts. Whether a particular write is
*permitted* is a different question, answered by ``artifact_guard``.

Every driver used to hardcode ``<repo>/artifacts``, which left no way to run one
without aiming it at the real evidence tree. The only safe way to run a driver
for a quick check was a detached git worktree — which works, but relies on the
person remembering to do it, and nothing about the code suggested they should.

Setting ``VALIDATION_ARTIFACT_ROOT`` redirects every driver's output somewhere
harmless::

    VALIDATION_ARTIFACT_ROOT=/tmp/scratch python drivers/mvnmle/fp32_gate.py

The two mechanisms compose. With the override pointed at scratch, the guard never
fires, because nothing there is tracked by git — so an exploratory run is
frictionless. Without it, a driver aims at the real tree and the guard refuses to
overwrite anything blessed. The easy path is the safe one, and the destructive
path needs a deliberate act.

The repo root is passed in rather than discovered here: a module that silently
locates the repo is a module that behaves differently depending on where it was
imported from.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set to redirect all generated evidence somewhere other than <repo>/artifacts.
ARTIFACT_ROOT_ENV = "VALIDATION_ARTIFACT_ROOT"


def artifact_root(repo: str | Path) -> Path:
    """Return the root under which artifacts are written.

    ``<repo>/artifacts`` by default; the value of ``VALIDATION_ARTIFACT_ROOT``
    when that is set to a non-empty value.

    The directory is not created and is not required to exist — callers already
    ``mkdir(parents=True)`` at write time, and creating it here would leave empty
    directories behind on any run that failed before writing.
    """
    override = os.environ.get(ARTIFACT_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    return Path(repo) / "artifacts"
