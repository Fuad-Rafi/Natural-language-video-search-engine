"""Configuration loading and path resolution.

The YAML file is the single source of defaults; command-line flags override it.
All relative paths are resolved against the repository root so scripts behave
the same regardless of the current working directory.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

_MISSING = object()


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


class Config:
    """Read-only view over the parsed configuration."""

    def __init__(
        self,
        data: Mapping[str, Any],
        path: Path = DEFAULT_CONFIG_PATH,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self._data = copy.deepcopy(dict(data))
        self.path = Path(path)
        self.repo_root = Path(repo_root)

    # -- generic access -------------------------------------------------

    def get(self, dotted: str, default: Any = None) -> Any:
        """Return a nested value, e.g. ``get("video.sample_fps")``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"missing required config key: {dotted!r}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"missing config section: {name!r}")
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section {name!r} must be a mapping")
        return copy.deepcopy(dict(value))

    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- paths ----------------------------------------------------------

    def path_for(self, dotted: str, default: Optional[str] = None) -> Path:
        """Resolve a configured path against the repository root."""
        value = self.get(dotted, default)
        if value is None:
            raise ConfigError(f"missing path config: {dotted!r}")
        return self.resolve(value)

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.repo_root / candidate

    @property
    def data_dir(self) -> Path:
        return self.path_for("paths.data_dir", "data")

    @property
    def logs_dir(self) -> Path:
        return self.path_for("paths.logs_dir", "logs")

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def ocr_dir(self) -> Path:
        return self.data_dir / "ocr"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    # -- overrides ------------------------------------------------------

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        if not overrides:
            return self
        return Config(_deep_merge(self._data, overrides), self.path, self.repo_root)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(path={str(self.path)!r})"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """Load a YAML config, applying optional nested overrides."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"config root must be a mapping: {config_path}")
    config = Config(dict(data), config_path, REPO_ROOT)
    return config.with_overrides(overrides or {})
