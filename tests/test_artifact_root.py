"""Tests for artifact-root resolution and its composition with the guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SHARED = Path(__file__).resolve().parent.parent / "drivers" / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import artifact_guard  # noqa: E402
import artifact_root  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(artifact_root.ARTIFACT_ROOT_ENV, raising=False)
    monkeypatch.delenv(artifact_guard.ALLOW_OVERWRITE_ENV, raising=False)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def test_defaults_to_repo_artifacts(tmp_path):
    assert artifact_root.artifact_root(tmp_path) == tmp_path / "artifacts"


def test_env_override_wins(tmp_path, monkeypatch):
    elsewhere = tmp_path / "scratch"
    monkeypatch.setenv(artifact_root.ARTIFACT_ROOT_ENV, str(elsewhere))
    assert artifact_root.artifact_root(tmp_path) == elsewhere


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_override_is_ignored(tmp_path, monkeypatch, value):
    """An empty variable must not redirect artifacts to the current directory."""
    monkeypatch.setenv(artifact_root.ARTIFACT_ROOT_ENV, value)
    assert artifact_root.artifact_root(tmp_path) == tmp_path / "artifacts"


def test_accepts_str_repo(tmp_path):
    assert artifact_root.artifact_root(str(tmp_path)) == tmp_path / "artifacts"


def test_does_not_create_the_directory(tmp_path):
    """Resolving a path must not leave an empty directory behind."""
    root = artifact_root.artifact_root(tmp_path)
    assert not root.exists()


# --------------------------------------------------------------------------- #
# Composition with the guard — the property the pair exists for
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo_with_blessed(tmp_path):
    r = tmp_path / "repo"
    (r / "artifacts" / "mod").mkdir(parents=True)
    _git(r.parent, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    (r / "artifacts" / "mod" / "run.json").write_text("{}")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "bless")
    artifact_guard._repo_root.cache_clear()
    artifact_guard._tracked_files.cache_clear()
    return r


def test_without_override_the_guard_protects_blessed_evidence(repo_with_blessed):
    target = artifact_root.artifact_root(repo_with_blessed) / "mod" / "run.json"
    with pytest.raises(RuntimeError, match="refusing to overwrite committed"):
        artifact_guard.guard_artifact_path(target)


def test_with_override_the_same_write_is_allowed(repo_with_blessed, tmp_path,
                                                 monkeypatch):
    """Redirected output is untracked, so the guard has nothing to object to.

    This is the whole design: an exploratory run is frictionless once pointed at
    scratch, and only an unredirected run can threaten real evidence.
    """
    scratch = tmp_path / "scratch"
    monkeypatch.setenv(artifact_root.ARTIFACT_ROOT_ENV, str(scratch))
    artifact_guard._repo_root.cache_clear()
    artifact_guard._tracked_files.cache_clear()

    target = artifact_root.artifact_root(repo_with_blessed) / "mod" / "run.json"
    assert scratch in target.parents
    assert artifact_guard.guard_artifact_path(target) == target


def test_override_does_not_disable_the_guard_elsewhere(repo_with_blessed,
                                                       tmp_path, monkeypatch):
    """Setting the override must not become a blanket permission slip."""
    monkeypatch.setenv(artifact_root.ARTIFACT_ROOT_ENV, str(tmp_path / "scratch"))
    artifact_guard._repo_root.cache_clear()
    artifact_guard._tracked_files.cache_clear()
    blessed = repo_with_blessed / "artifacts" / "mod" / "run.json"
    with pytest.raises(RuntimeError):
        artifact_guard.guard_artifact_path(blessed)
