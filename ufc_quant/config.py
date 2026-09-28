"""Central configuration loading and path resolution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Resolved directories for one dataset kind (real or synthetic)."""

    raw: Path
    processed: Path
    artifacts: Path
    reports: Path
    dataset_kind: str  # "real" or "synthetic"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.artifacts, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_paths(cfg: dict[str, Any], synthetic: bool = False) -> Paths:
    p = cfg["paths"]
    if synthetic:
        root = REPO_ROOT / p["synthetic_root"]
        return Paths(root / "raw", root / "processed", root / "artifacts", root / "reports", "synthetic")
    return Paths(
        REPO_ROOT / p["raw_dir"],
        REPO_ROOT / p["processed_dir"],
        REPO_ROOT / p["artifacts_dir"],
        REPO_ROOT / p["reports_dir"],
        "real",
    )


def override(cfg: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Return a deep copy of cfg with one dotted key replaced (used by sensitivity analyses)."""
    out = copy.deepcopy(cfg)
    node = out
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return out
