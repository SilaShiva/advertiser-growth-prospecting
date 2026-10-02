"""Config loader: single entry point every stage uses.

Pattern: no stage hard-codes a path, threshold, or hyperparameter.
Everything lives in configs/config.yaml and is loaded once through
`load_config()`, which resolves repo-relative paths to absolute ones so
stages can be invoked from any working directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "config.yaml"


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Turn repo-relative path entries into absolute Paths, in place."""
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else (_REPO_ROOT / p)
    return cfg


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config.yaml and resolve relative paths. Fails fast on typos."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "project" not in cfg:
        raise ValueError(f"malformed config: {cfg_path} (missing 'project' section)")

    required_sections = ("project", "paths", "generation", "validation",
                         "features", "labels", "splits", "training", "inference")
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(f"config missing required sections: {missing}")

    return _resolve_paths(cfg)


def repo_root() -> Path:
    return _REPO_ROOT
