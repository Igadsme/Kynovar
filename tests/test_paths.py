"""Project paths stay on the repository drive unless a data root is set."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kynovar.utils.paths import (
    PathConfigError,
    ensure_within_project,
    find_repo_root,
    load_paths,
    validate_startup,
)

_CLEARED = (
    "KYNOVAR_ROOT",
    "KYNOVAR_DATA",
    "KYNOVAR_DATA_ROOT",
    "KYNOVAR_CHECKPOINTS",
    "KYNOVAR_RESULTS",
    "KYNOVAR_LOGS",
)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _CLEARED:
        monkeypatch.delenv(key, raising=False)


def test_defaults_live_under_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    paths = load_paths()
    root = find_repo_root()
    assert paths.root == root
    assert paths.data == root / "datasets"
    assert paths.checkpoints == root / "checkpoints"
    assert paths.results == root / "results"
    assert paths.logs == root / "logs"
    assert paths.cache == root / ".cache"
    assert paths.hf_home == root / ".cache" / "huggingface"
    assert paths.torch_home == root / ".cache" / "torch"
    source = (root / "kynovar" / "utils" / "paths.py").read_text(encoding="utf-8")
    assert "T7 Shield" not in source


def test_relative_override_stays_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("KYNOVAR_DATA", "datasets")
    assert load_paths().data == find_repo_root() / "datasets"


def test_paths_outside_the_repository_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("KYNOVAR_DATA", "/tmp/kynovar-outside")
    with pytest.raises(PathConfigError):
        load_paths()


def test_protected_directories_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("KYNOVAR_CHECKPOINTS", "/tmp")
    with pytest.raises(PathConfigError):
        load_paths()


def test_root_must_be_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("KYNOVAR_ROOT", "/tmp")
    with pytest.raises(PathConfigError):
        load_paths()


def test_data_directory_cannot_be_the_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("KYNOVAR_DATA", str(find_repo_root()))
    with pytest.raises(PathConfigError):
        load_paths()


def test_explicit_data_root_can_hold_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("KYNOVAR_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("KYNOVAR_DATA", str(tmp_path / "datasets"))
    paths = load_paths()
    assert paths.data == (tmp_path / "datasets").resolve()


def test_startup_creates_writable_caches_inside_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("HF_HOME", "/tmp/should-not-be-used")
    monkeypatch.setenv("TORCH_HOME", "/tmp/should-not-be-used")
    paths = validate_startup()
    assert paths.hf_home.is_dir()
    assert paths.torch_home.is_dir()
    assert not (paths.hf_home / ".kynovar_write_probe").exists()
    assert os.environ["HF_HOME"] == str(paths.hf_home)
    assert os.environ["TORCH_HOME"] == str(paths.torch_home)
    assert os.environ["XDG_CACHE_HOME"] == str(paths.cache)
    assert Path(os.environ["HF_HOME"]).is_relative_to(paths.root)
    assert Path(os.environ["KYNOVAR_DATA"]).is_relative_to(paths.root)


def test_output_paths_cannot_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    paths = load_paths()
    escaped = paths.root.parent / "not-kynovar.json"
    with pytest.raises(PathConfigError):
        ensure_within_project(escaped, paths)
    resolved = ensure_within_project(Path("results/demo.json"), paths)
    assert resolved == paths.results / "demo.json"
