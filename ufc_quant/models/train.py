"""Walk-forward model selection, calibration and final training.

Protocol
1. Validation (years in `walk_forward_years`, all < test_start): for each year Y, fit on
   [train_start, Y) and predict Y. Hyper-parameters are chosen on the pooled out-of-fold
   log-loss of these years only.
2. Calibration (temperature scaling): for validation year Y it is fitted on the out-of-fold
   predictions of years < Y; for the test it is fitted on all validation years.
3. Test: each model family is refitted once on [train_start, test_start) with its selected
   hyper-parameters and predicts the untouched test period. The test is never used to choose.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_quant.evaluation.splits import assert_event_disjoint, final_split, walk_forward_folds
from ufc_quant.models.core import TemperatureCalibrator, log_loss, make_model

log = logging.getLogger(__name__)

META_COLS = ["fight_id", "event_id", "event_date", "fighter_a_id", "fighter_b_id", "result_a", "y"]


def candidate_grid(cfg: dict) -> dict[str, list[dict]]:
    m = cfg["models"]
    return {
        "baseline": [{}],
        "elo": [{}],
        "logistic": [{"C": c} for c in m["logistic"]["C_grid"]],
        "gbm": list(m["gbm"]["param_grid"]),
    }


def run_walk_forward(feats: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    seed = cfg["project"]["random_seed"]
    rows, cv_scores = [], {}
    grid = candidate_grid(cfg)
    folds = list(walk_forward_folds(feats, cfg, include_undecided=True))
    for family, cands in grid.items():
        for i, params in enumerate(cands):
            preds = []
            for year, tr, va in folds:
                assert_event_disjoint(tr, va)
                model = make_model(family, params, seed).fit(tr)
                preds.append(va[META_COLS].assign(fold=year, p_a=model.predict_proba_a(va)))
            pr = pd.concat(preds)
            dec = pr[pr["y"].notna()]
            cv_scores[(family, i)] = log_loss(dec["y"], dec["p_a"])
            rows.append(pr.assign(model=family, cand=i))
            log.info("%s %s pooled OOF log-loss %.4f", family, params, cv_scores[(family, i)])
    oof = pd.concat(rows, ignore_index=True)
    selection = {}
    for family, cands in grid.items():
        best = min(range(len(cands)), key=lambda i: cv_scores[(family, i)])
        selection[family] = {"params": cands[best], "cand": best,
                             "cv_log_loss": {json.dumps(c): cv_scores[(family, i)] for i, c in enumerate(cands)}}
    return oof, selection


def expanding_calibration(oof_family: pd.DataFrame) -> pd.Series:
    """Calibrated OOF predictions: year Y uses a temperature fitted on years < Y (Y0 left raw)."""
    out = pd.Series(np.nan, index=oof_family.index)
    decided = oof_family["y"].notna()
    for year in sorted(oof_family["fold"].unique()):
        cur = oof_family["fold"] == year
        past = (oof_family["fold"] < year) & decided
        if past.sum() == 0:
            out[cur] = oof_family.loc[cur, "p_a"]
            continue
        cal = TemperatureCalibrator().fit(oof_family.loc[past, "p_a"], oof_family.loc[past, "y"])
        out[cur] = cal.transform(oof_family.loc[cur, "p_a"])
    return out


def train_all(feats: pd.DataFrame, cfg: dict, artifacts_dir: Path, data_meta: dict) -> dict:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["project"]["random_seed"]
    oof_all, selection = run_walk_forward(feats, cfg)

    # keep the selected candidate of each family, add calibrated variants
    val_frames, calibrators = [], {}
    for family, sel in selection.items():
        fam = oof_all[(oof_all["model"] == family) & (oof_all["cand"] == sel["cand"])].copy()
        val_frames.append(fam)
        if cfg["models"]["calibration"]["enabled"] and family in ("logistic", "gbm"):
            cal_oof = fam.assign(p_a=expanding_calibration(fam).values, model=f"{family}_cal")
            val_frames.append(cal_oof)
            dec = fam[fam["y"].notna()]
            calibrators[family] = TemperatureCalibrator().fit(dec["p_a"], dec["y"])
    val = pd.concat(val_frames, ignore_index=True).assign(split="validation")

    # final fit on all pre-test data, single prediction of the test period
    tr, te = final_split(feats, cfg, include_undecided=True)
    assert_event_disjoint(tr, te)
    test_frames, final_models = [], {}
    for family, sel in selection.items():
        model = make_model(family, sel["params"], seed).fit(tr)
        final_models[family] = model
        p = model.predict_proba_a(te)
        test_frames.append(te[META_COLS].assign(fold=-1, p_a=p, model=family))
        if family in calibrators:
            test_frames.append(te[META_COLS].assign(fold=-1, p_a=calibrators[family].transform(p), model=f"{family}_cal"))
    test = pd.concat(test_frames, ignore_index=True).assign(split="test")

    preds = pd.concat([val, test], ignore_index=True).drop(columns=["cand"], errors="ignore")
    # model used by the backtest: best pooled validation log-loss among the sport models
    vdec = val[val["y"].notna() & val["model"].isin(["elo", "logistic", "logistic_cal", "gbm", "gbm_cal"])]
    val_ll = vdec.groupby("model").apply(lambda g: log_loss(g["y"], g["p_a"]), include_groups=False)
    betting_model = str(val_ll.idxmin())
    preds.to_parquet(artifacts_dir / "predictions.parquet", index=False)
    with open(artifacts_dir / "final_models.pkl", "wb") as fh:
        pickle.dump({"models": final_models, "calibrators": calibrators}, fh)

    meta = {
        "model_version": cfg["project"]["model_version"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_kind": data_meta.get("dataset_kind"),
        "data_max_date": data_meta.get("data_max_date"),
        "train_start": cfg["splits"]["train_start"],
        "test_start": cfg["splits"]["test_start"],
        "walk_forward_years": cfg["splits"]["walk_forward_years"],
        "n_train_final": int(len(tr)), "n_test_decided": int(te["y"].notna().sum()),
        "n_test_predicted_incl_draw_nc": int(len(te)),
        "elo_k": data_meta.get("elo_k"),
        "selection": selection,
        "temperatures": {k: v.T for k, v in calibrators.items()},
        "validation_log_loss_selected": val_ll.to_dict(),
        "betting_model": betting_model,
    }
    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta
