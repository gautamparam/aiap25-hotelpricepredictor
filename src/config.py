"""Configuration loading.

The pipeline is driven by a YAML file so algorithms, hyper-parameters and
preprocessing choices can be swapped without touching code. Any key may be
overridden from the command line (``--set model.name=xgboost``) or by an
environment variable (``TI_DB_PATH``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root, resolved relative to this file so the code works no matter
# which directory the notebook or `run.sh` is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The assessment brief is ambiguous here: it says to place `hotel.db` in a
# `data` folder, but also to retrieve it via the relative path `hotel.db`.
# We default to `data/hotel.db` and fall back to the repository root, so both
# readings work without configuration.
DEFAULT_DB_CANDIDATES = (
    PROJECT_ROOT / "data" / "hotel.db",
    PROJECT_ROOT / "hotel.db",
)


def resolve_db_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate `hotel.db`, preferring an explicit path, then $TI_DB_PATH.

    Raises FileNotFoundError listing every location tried, because a missing
    database is the single most likely setup failure for an assessor.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if env := os.environ.get("TI_DB_PATH"):
        candidates.append(Path(env))
    candidates.extend(DEFAULT_DB_CANDIDATES)

    for path in candidates:
        if path.is_file():
            return path
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find hotel.db. Tried:\n  {tried}\n"
        "Download it from the assessment URL and place it at data/hotel.db."
    )


@dataclass
class Config:
    """Typed view over the YAML config, with sane defaults for every key."""

    db_path: str | None = None
    random_seed: int = 42
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
        raw: dict[str, Any] = {}
        if cfg_path.is_file():
            raw = yaml.safe_load(cfg_path.read_text()) or {}
        return cls(
            db_path=raw.get("data", {}).get("db_path"),
            random_seed=raw.get("random_seed", 42),
            raw=raw,
        )

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested key, e.g. ``cfg.get("model.params.max_depth")``."""
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
