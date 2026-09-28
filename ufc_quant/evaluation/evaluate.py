"""Evaluation step: metrics, calibration, market comparison, interpretation."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_quant.evaluation.market import (encompassing_regression, event_bootstrap_delta, market_probabilities,
                                         source_consistency)
from ufc_quant.evaluation.metrics import calibration_table, expected_calibration_error, score
from ufc_quant.evaluation.splits import walk_forward_folds
from ufc_quant.models.core import log_loss, make_model

log = logging.getLogger(__name__)


def permutation_importance_sym(model, df: pd.DataFrame, features: list[str], n_repeats: int = 5,
                               seed: int = 0) -> pd.DataFrame:
    """Increase in log-loss when one feature is permuted (symmetrised predictor)."""
    rng = np.random.default_rng(seed)
    base = log_loss(df["y"], model.predict_proba_a(df))
    rows = []
    for f in features:
        incs = []
        for _ in range(n_repeats):
            d = df.copy()
            d[f] = rng.permutation(d[f].values)
            incs.append(log_loss(d["y"], model.predict_proba_a(d)) - base)
        rows.append({"feature": f, "importance_mean": float(np.mean(incs)), "importance_std": float(np.std(incs))})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def evaluate(cfg: dict, processed_dir: Path, artifacts_dir: Path) -> dict:
    processed_dir, artifacts_dir = Path(processed_dir), Path(artifacts_dir)
    preds_all = pd.read_parquet(artifacts_dir / "predictions.parquet")
    preds = preds_all[preds_all["y"].notna()]  # scoring only on decided fights
    feats = pd.read_parquet(processed_dir / "features.parquet")
    fights = pd.read_parquet(processed_dir / "fights.parquet")
    odds_path = processed_dir / "odds.parquet"
    odds = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()
    meta = json.loads((artifacts_dir / "model_meta.json").read_text())
    seed = cfg["project"]["random_seed"]
    n_boot = cfg["backtest"]["bootstrap"]["n_resamples"]
    out: dict = {"model_version": meta["model_version"], "dataset_kind": meta.get("dataset_kind")}

    # 1. probability quality -------------------------------------------------
    metrics, calib = [], []
    for (split, model), g in preds.groupby(["split", "model"]):
        s = score(g["y"], g["p_a"])
        s.update(split=split, model=model, ece=expected_calibration_error(g["y"], g["p_a"]),
                 date_min=str(g["event_date"].min().date()), date_max=str(g["event_date"].max().date()))
        metrics.append(s)
        calib.append(calibration_table(g["y"], g["p_a"]).assign(split=split, model=model))
    metrics_df = pd.DataFrame(metrics)
    by_year = [dict(score(g["y"], g["p_a"]), model=m, year=int(pd.Timestamp(g["event_date"].iloc[0]).year))
               for (m, _y), g in preds.assign(_y=preds["event_date"].dt.year).groupby(["model", "_y"])]
    out["metrics"] = metrics
    out["coverage"] = {
        "n_fights_total": int(len(feats)),
        "n_decided": int(feats["y"].notna().sum()),
        "n_draw": int((feats["result_a"] == "draw").sum()),
        "n_no_contest": int((feats["result_a"] == "no_contest").sum()),
        "n_ambiguous_ids_excluded": int((feats["id_ambiguous"] & feats["y"].notna()).sum()),
        "n_validation": int(preds.query("split == 'validation' and model == 'baseline'").shape[0]),
        "n_test": int(preds.query("split == 'test' and model == 'baseline'").shape[0]),
        "feature_missing_rate_eval": feats[feats["event_date"] >= cfg["splits"]["train_start"]]
        .filter(like="diff_").isna().mean().round(4).to_dict(),
    }

    # 2. market comparison ---------------------------------------------------
    mkt = market_probabilities(odds, fights, feats, cfg)
    out["market_available"] = bool(len(mkt))
    comp_rows, enc, boot, mkt_calib = [], {}, {}, []
    if len(mkt):
        out["reference_bookmaker"] = cfg["odds"]["reference_bookmaker"]
        out["odds_snapshot_precision"] = mkt["snapshot_precision"].value_counts().to_dict()
        out["overround"] = {"mean": float(mkt["overround"].mean()), "median": float(mkt["overround"].median()),
                            "p05": float(mkt["overround"].quantile(0.05)), "p95": float(mkt["overround"].quantile(0.95))}
        wide = preds.pivot_table(index=["fight_id", "event_id", "split", "y"], columns="model", values="p_a").reset_index()
        wide = wide.merge(mkt[["fight_id", "p_market_a"]], on="fight_id", how="left")
        for split, g in wide.groupby("split"):
            n_all = len(g)
            gm = g.dropna(subset=["p_market_a"])
            out.setdefault("market_coverage", {})[split] = {"n_fights": n_all, "n_with_odds": int(len(gm)),
                                                            "share": float(len(gm) / n_all) if n_all else 0.0}
            if len(gm) < 30:
                continue
            comp_rows.append(dict(score(gm["y"], gm["p_market_a"]), split=split, model="market"))
            mkt_calib.append(calibration_table(gm["y"], gm["p_market_a"]).assign(split=split, model="market"))
            for m in [c for c in preds["model"].unique()]:
                comp_rows.append(dict(score(gm["y"], gm[m]), split=split, model=m))
                if m in ("elo", "logistic", "gbm", "logistic_cal", "gbm_cal"):
                    enc[f"{split}:{m}"] = encompassing_regression(gm["y"], gm["p_market_a"], gm[m])
                    boot[f"{split}:{m}"] = event_bootstrap_delta(gm, m, "p_market_a", n=n_boot, seed=seed)
        out["market_comparison_same_fights"] = comp_rows
        out["encompassing_regression"] = enc
        out["bootstrap_logloss_model_minus_market"] = boot
    out["odds_source_consistency"] = source_consistency(odds, feats) if len(odds) else {}

    # 3. interpretation (pre-test data only) ---------------------------------
    with open(artifacts_dir / "final_models.pkl", "rb") as fh:
        final = pickle.load(fh)
    coefs = final["models"]["logistic"].coefficients()
    out["logistic_coefficients_std"] = coefs.round(4).to_dict()
    folds = list(walk_forward_folds(feats, cfg))
    if folds:
        year, tr, va = folds[-1]  # decided fights only (include_undecided=False)
        imps = []
        for fam in ("logistic", "gbm"):
            m = make_model(fam, meta["selection"][fam]["params"], seed).fit(tr)
            imps.append(permutation_importance_sym(m, va, m.features, seed=seed).assign(model=fam, eval_year=year))
        pd.concat(imps).to_parquet(artifacts_dir / "permutation_importance.parquet", index=False)

    metrics_df.to_parquet(artifacts_dir / "metrics.parquet", index=False)
    pd.DataFrame(by_year).to_parquet(artifacts_dir / "metrics_by_year.parquet", index=False)
    pd.concat(calib + mkt_calib).to_parquet(artifacts_dir / "calibration.parquet", index=False)
    if len(mkt):
        mkt.to_parquet(artifacts_dir / "market_probabilities.parquet", index=False)
    (artifacts_dir / "evaluation.json").write_text(json.dumps(out, indent=2, default=str))
    return out
