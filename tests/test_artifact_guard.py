"""Tests for the blessed-evidence overwrite guard.

Builds a real throwaway git repo under ``tmp_path`` for each test, so the guard
is exercised against actual ``git ls-files`` output rather than a mock of it —
the whole behaviour hinges on that call being right.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SHARED = Path(__file__).resolve().parent.parent / "drivers" / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import artifact_guard  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo with one COMMITTED artifact and one untracked file."""
    r = tmp_path / "repo"
    (r / "artifacts").mkdir(parents=True)
    _git(r.parent, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")

    (r / "artifacts" / "blessed.json").write_text('{"blessed": true}')
    _git(r, "add", "artifacts/blessed.json")
    _git(r, "commit", "-qm", "bless")

    (r / "artifacts" / "fresh.json").write_text('{"fresh": true}')

    monkeypatch.delenv(artifact_guard.ALLOW_OVERWRITE_ENV, raising=False)
    artifact_guard._repo_root.cache_clear()
    artifact_guard._tracked_files.cache_clear()
    return r


# --------------------------------------------------------------------------- #
# Normal cases — ordinary work must not be obstructed
# --------------------------------------------------------------------------- #
def test_untracked_file_passes(repo):
    p = repo / "artifacts" / "fresh.json"
    assert artifact_guard.guard_artifact_path(p) == p


def test_nonexistent_file_passes(repo):
    p = repo / "artifacts" / "brand_new.json"
    assert artifact_guard.guard_artifact_path(p) == p


def test_path_outside_any_repo_passes(tmp_path, monkeypatch):
    monkeypatch.delenv(artifact_guard.ALLOW_OVERWRITE_ENV, raising=False)
    artifact_guard._repo_root.cache_clear()
    outside = tmp_path / "loose" / "x.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("{}")
    assert artifact_guard.guard_artifact_path(outside) == outside


# --------------------------------------------------------------------------- #
# The case the guard exists for
# --------------------------------------------------------------------------- #
def test_committed_evidence_is_refused(repo):
    p = repo / "artifacts" / "blessed.json"
    with pytest.raises(RuntimeError, match="refusing to overwrite committed"):
        artifact_guard.guard_artifact_path(p)


def test_refusal_names_the_escape_hatch(repo):
    """A guard that blocks without saying how to proceed is a dead end."""
    with pytest.raises(RuntimeError) as exc:
        artifact_guard.guard_artifact_path(repo / "artifacts" / "blessed.json")
    assert artifact_guard.ALLOW_OVERWRITE_ENV in str(exc.value)


def test_committed_evidence_is_untouched_after_refusal(repo):
    """Refusing must not have partially written or truncated the target."""
    p = repo / "artifacts" / "blessed.json"
    before = p.read_text()
    with pytest.raises(RuntimeError):
        artifact_guard.guard_artifact_path(p)
    assert p.read_text() == before


# --------------------------------------------------------------------------- #
# The escape hatch — a real re-validation must still be possible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["1", "yes", "true"])
def test_env_var_permits_overwrite(repo, monkeypatch, value):
    monkeypatch.setenv(artifact_guard.ALLOW_OVERWRITE_ENV, value)
    p = repo / "artifacts" / "blessed.json"
    assert artifact_guard.guard_artifact_path(p) == p


@pytest.mark.parametrize("value", ["", "0"])
def test_empty_or_zero_does_not_permit_overwrite(repo, monkeypatch, value):
    """An unset-like value must not read as permission."""
    monkeypatch.setenv(artifact_guard.ALLOW_OVERWRITE_ENV, value)
    with pytest.raises(RuntimeError):
        artifact_guard.guard_artifact_path(repo / "artifacts" / "blessed.json")


def test_is_tracked_distinguishes_the_two_files(repo):
    assert artifact_guard.is_tracked(repo / "artifacts" / "blessed.json") is True
    assert artifact_guard.is_tracked(repo / "artifacts" / "fresh.json") is False
