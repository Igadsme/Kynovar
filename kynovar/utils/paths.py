"""Resolve and validate project paths at startup.

Large artifacts stay under the repository root, or under an explicit
KYNOVAR_DATA_ROOT. Volume names are never hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROTECTED_EXACT = {
    Path("/"),
    Path("/Users"),
    Path("/Volumes"),
    Path("/System"),
    Path("/Library"),
    Path("/private"),
    Path("/tmp"),
    Path("/var"),
    Path("/opt"),
    Path("/usr"),
    Path("/bin"),
    Path("/Applications"),
}


class PathConfigError(RuntimeError):
    """A project path is missing, protected, or outside the allowed root."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    checkpoints: Path
    results: Path
    logs: Path
    cache: Path
    hf_home: Path
    torch_home: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "KYNOVAR_ROOT": str(self.root),
            "KYNOVAR_DATA": str(self.data),
            "KYNOVAR_CHECKPOINTS": str(self.checkpoints),
            "KYNOVAR_RESULTS": str(self.results),
            "KYNOVAR_LOGS": str(self.logs),
            "XDG_CACHE_HOME": str(self.cache),
            "HF_HOME": str(self.hf_home),
            "TORCH_HOME": str(self.torch_home),
        }


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents until a directory contains pyproject.toml and kynovar/."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "kynovar").is_dir():
            return candidate
    raise PathConfigError(
        "Could not locate the Kynovar repository root "
        "(expected pyproject.toml next to the kynovar package)."
    )


def _reject_protected(path: Path, label: str) -> None:
    resolved = path.resolve()
    home = Path.home().resolve()
    if resolved == home or resolved in _PROTECTED_EXACT or resolved == Path(resolved.anchor):
        raise PathConfigError(
            f"{label}={resolved} is a protected directory and cannot hold project data."
        )


def _require_inside(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    try:
        resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise PathConfigError(
            f"{label}={resolved} is outside the allowed root {parent_resolved}."
        ) from exc
    return resolved


def _resolve_configured(env_name: str, default: Path, root: Path) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        path = default
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_paths(start: Path | None = None) -> ProjectPaths:
    """Load path configuration. Does not create directories or mutate the environment."""
    env_root = os.environ.get("KYNOVAR_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / "pyproject.toml").is_file() or not (root / "kynovar").is_dir():
            raise PathConfigError(
                f"KYNOVAR_ROOT={root} does not contain the Kynovar project "
                "(pyproject.toml and kynovar/)."
            )
    else:
        root = find_repo_root(start)

    _reject_protected(root, "KYNOVAR_ROOT")

    data_root_raw = os.environ.get("KYNOVAR_DATA_ROOT", "").strip()
    allowed_parents = [root]
    if data_root_raw:
        data_root = Path(data_root_raw).expanduser()
        if not data_root.is_absolute():
            data_root = root / data_root
        data_root = data_root.resolve()
        _reject_protected(data_root, "KYNOVAR_DATA_ROOT")
        allowed_parents.append(data_root)

    def locate(env_name: str, default: Path) -> Path:
        path = _resolve_configured(env_name, default, root)
        _reject_protected(path, env_name)
        if path == root:
            raise PathConfigError(
                f"{env_name} must be a directory inside the repository, not the repository root."
            )
        for parent in allowed_parents:
            try:
                return _require_inside(path, parent, env_name)
            except PathConfigError:
                continue
        raise PathConfigError(
            f"{env_name}={path} must be inside the repository ({root})"
            + (
                f" or KYNOVAR_DATA_ROOT ({allowed_parents[-1]})"
                if len(allowed_parents) > 1
                else ". Set KYNOVAR_DATA_ROOT only when artifacts must live beside the repo."
            )
        )

    # Cache locations are project-owned. Ambient HF_HOME / TORCH_HOME /
    # XDG_CACHE_HOME values are not inputs; startup overwrites them for this
    # process so downloads cannot land in a home-directory cache.
    cache = (root / ".cache").resolve()
    return ProjectPaths(
        root=root,
        data=locate("KYNOVAR_DATA", root / "datasets"),
        checkpoints=locate("KYNOVAR_CHECKPOINTS", root / "checkpoints"),
        results=locate("KYNOVAR_RESULTS", root / "results"),
        logs=locate("KYNOVAR_LOGS", root / "logs"),
        cache=cache,
        hf_home=(cache / "huggingface").resolve(),
        torch_home=(cache / "torch").resolve(),
    )


def ensure_within_project(path: Path, paths: ProjectPaths) -> Path:
    """Resolve `path` and require it to stay inside the repository or data root."""
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = paths.root / candidate
    candidate = candidate.resolve()
    _reject_protected(candidate, "output path")
    parents = [paths.root, paths.data, paths.checkpoints, paths.results, paths.logs, paths.cache]
    for parent in parents:
        try:
            candidate.relative_to(parent.resolve())
            return candidate
        except ValueError:
            continue
    try:
        candidate.relative_to(paths.root.resolve())
        return candidate
    except ValueError as exc:
        raise PathConfigError(
            f"Refusing to write {candidate}, which is outside the Kynovar project root {paths.root}."
        ) from exc


def validate_startup(paths: ProjectPaths | None = None) -> ProjectPaths:
    """Create project directories, probe writability, and point caches at the project drive."""
    resolved = paths or load_paths()
    directories = [
        resolved.data,
        resolved.data / "raw",
        resolved.data / "generated",
        resolved.data / "processed",
        resolved.data / "test",
        resolved.checkpoints,
        resolved.results,
        resolved.logs,
        resolved.cache,
        resolved.hf_home,
        resolved.torch_home,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise PathConfigError(f"Could not create directory {directory}.")
        probe = directory / ".kynovar_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise PathConfigError(f"Directory {directory} is not writable.") from exc
    apply_runtime_env(resolved)
    return resolved


def apply_runtime_env(paths: ProjectPaths) -> None:
    """Point this process at the validated project paths and caches."""
    for key, value in paths.as_dict().items():
        os.environ[key] = value
