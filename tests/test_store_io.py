"""Tests for central dataset store resolution (``drivers/_shared/store_io.py``).

``store_io`` is the single place that knows where the dataset store lives, so it
is also the single point of failure for every driver's data load. Ten stale
copies of that search path previously existed across the drivers; these tests
exist so the consolidated one cannot rot silently.

Every test builds a fake store under ``tmp_path`` and monkeypatches the mirror
list, so nothing here depends on the real store being present — the suite passes
on a machine that has never seen the data (Forge, CI, a fresh clone).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

_SHARED = Path(__file__).resolve().parent.parent / "drivers" / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import store_io  # noqa: E402


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    """A store root containing ``pystatistics/demo.h5``, with no env vars set."""
    root = tmp_path / "datasets"
    (root / store_io.DEFAULT_NAMESPACE).mkdir(parents=True)
    (root / store_io.DEFAULT_NAMESPACE / "demo.h5").write_bytes(b"not really hdf5")
    monkeypatch.delenv(store_io.STORE_ROOT_ENV, raising=False)
    monkeypatch.delenv(store_io.STORE_ROOT_ENV_LEGACY, raising=False)
    monkeypatch.setattr(store_io, "_STORE_MIRRORS", ())
    return root


# --------------------------------------------------------------------------- #
# Normal cases
# --------------------------------------------------------------------------- #
def test_root_from_current_env(fake_store, monkeypatch):
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    assert store_io.store_root() == fake_store


def test_root_falls_back_to_mirror_when_env_unset(fake_store, monkeypatch):
    monkeypatch.setattr(store_io, "_STORE_MIRRORS", (fake_store,))
    assert store_io.store_root() == fake_store


def test_h5_path_is_namespaced(fake_store, monkeypatch):
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    got = store_io.store_h5_path("demo")
    assert got == fake_store / store_io.DEFAULT_NAMESPACE / "demo.h5"
    assert got.is_file()


def test_h5_path_honors_explicit_namespace(fake_store, monkeypatch):
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    (fake_store / "lacuna").mkdir()
    (fake_store / "lacuna" / "demo.h5").write_bytes(b"other namespace")
    got = store_io.store_h5_path("demo", namespace="lacuna")
    assert got == fake_store / "lacuna" / "demo.h5"


# --------------------------------------------------------------------------- #
# Deprecated alias
# --------------------------------------------------------------------------- #
def test_legacy_env_still_resolves_but_warns(fake_store, monkeypatch):
    monkeypatch.setenv(store_io.STORE_ROOT_ENV_LEGACY, str(fake_store))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert store_io.store_root() == fake_store
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_both_env_vars_agreeing_is_fine(fake_store, monkeypatch):
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    monkeypatch.setenv(store_io.STORE_ROOT_ENV_LEGACY, str(fake_store))
    assert store_io.store_root() == fake_store


def test_current_env_takes_precedence_over_legacy_when_equal(fake_store, monkeypatch):
    # Trailing slash must not read as disagreement — same directory, same result.
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    monkeypatch.setenv(store_io.STORE_ROOT_ENV_LEGACY, str(fake_store) + "/")
    assert store_io.store_root() == fake_store


# --------------------------------------------------------------------------- #
# Failure cases — all must be loud (Rule 1)
# --------------------------------------------------------------------------- #
def test_conflicting_env_vars_fail_loud(fake_store, monkeypatch):
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    monkeypatch.setenv(store_io.STORE_ROOT_ENV_LEGACY, str(fake_store / "elsewhere"))
    with pytest.raises(RuntimeError, match="disagree"):
        store_io.store_root()


def test_missing_root_fails_loud_and_lists_paths(fake_store, monkeypatch):
    missing = fake_store / "nope"
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(missing))
    with pytest.raises(FileNotFoundError) as exc:
        store_io.store_root()
    # The error must be actionable: it names what was tried.
    assert str(missing) in str(exc.value)
    assert store_io.STORE_ROOT_ENV in str(exc.value)


def test_missing_dataset_fails_loud(fake_store, monkeypatch):
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    with pytest.raises(FileNotFoundError, match="absent_dataset"):
        store_io.store_h5_path("absent_dataset")


def test_never_returns_a_path_that_does_not_exist(fake_store, monkeypatch):
    """The contract is 'a usable path or an exception' — never a hopeful path."""
    monkeypatch.setenv(store_io.STORE_ROOT_ENV, str(fake_store))
    try:
        got = store_io.store_h5_path("absent_dataset")
    except FileNotFoundError:
        return
    pytest.fail(f"returned non-existent path instead of raising: {got}")
