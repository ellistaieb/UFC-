"""Pipeline steps shared by the CLI and the tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ufc_quant.config import Paths
from ufc_quant.features.elo import tune_elo_k
from ufc_quant.features.engine import build_features

log = logging.getLogger(__name__)


def load_processed(paths: Paths) -> dict[str, pd.DataFrame]:
    p = paths.processed
    out = {name: pd.read_parquet(p / f"{name}.parquet") for name in ("fights", "fight_stats", "fighters")}
    odds_path = p / "odds.parquet"
    out["odds"] = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()
    return out


def dataset_meta(paths: Paths) -> dict:
    f = paths.processed / "dataset_meta.json"
    return json.loads(f.read_text()) if f.exists() else {"dataset_kind": paths.dataset_kind}


def step_features(cfg: dict, paths: Paths) -> dict:
    d = load_processed(paths)
    fights = d["fights"]
    ecfg = cfg["features"]["elo"]
    years = cfg["splits"]["walk_forward_years"]
    # K is tuned on validation years only (strictly before the test period)
    k, k_scores = tune_elo_k(fights, ecfg["k_grid"], f"{min(years)}-01-01", cfg["splits"]["test_start"],
                             ecfg["initial_rating"], ecfg["draw_score"])
    feats = build_features(fights, d["fight_stats"], d["fighters"], cfg, elo_k=k)
    feats.to_parquet(paths.processed / "features.parquet", index=False)
    meta = dataset_meta(paths)
    meta.update({
        "dataset_kind": paths.dataset_kind,
        "data_min_date": str(fights["event_date"].min().date()),
        "data_max_date": str(fights["event_date"].max().date()),
        "elo_k": k, "elo_k_validation_log_loss": k_scores,
        "n_feature_rows": int(len(feats)),
    })
    (paths.processed / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("features: %d rows, Elo K=%s", len(feats), k)
    return meta
