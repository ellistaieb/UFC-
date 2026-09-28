"""Backtest step: candidates = model probabilities + pre-decision odds, three strategies,
sensitivity analyses and event bootstrap. Strategy parameters are fixed in the config
BEFORE looking at the test period; sensitivity tables are descriptive only."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ufc_quant.backtest.engine import BacktestParams, bootstrap_unit_roi, by_period, run_backtest, summarize
from ufc_quant.evaluation.market import encompassing_regression, market_probabilities
from ufc_quant.models.core import logit, sigmoid

log = logging.getLogger(__name__)
STRATEGIES = ["flat", "fixed_fraction", "quarter_kelly"]


def build_candidates(preds_all: pd.DataFrame, mkt: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    p = preds_all[(preds_all["model"] == model) & (preds_all["split"] == split)]
    return p.merge(mkt[["fight_id", "odds_a", "odds_b", "p_market_a", "snapshot_precision"]], on="fight_id", how="left")


def blend_predictions(preds_all: pd.DataFrame, mkt: pd.DataFrame, model: str) -> pd.DataFrame:
    """Secondary analysis: p_blend = sigmoid(b_mkt * logit(p_market) + b_mod * logit(p_model)).

    Coefficients come from the encompassing regression fitted ONLY on earlier periods:
    validation year Y uses validation years < Y (the first year gets no prediction);
    the test period uses all validation years. Uses odds as an input, unlike the sport models.
    """
    base = preds_all[preds_all["model"] == model].merge(mkt[["fight_id", "p_market_a"]], on="fight_id", how="inner")
    val = base[base["split"] == "validation"]
    out, coefs = [], {}
    for year in sorted(val["fold"].unique()):
        past = val[(val["fold"] < year) & val["y"].notna()]
        if len(past) < 200:
            continue
        r = encompassing_regression(past["y"], past["p_market_a"], past["p_a"])
        cur = val[val["fold"] == year]
        coefs[int(year)] = (r["beta_market"], r["beta_model"])
        out.append(cur.assign(p_a=sigmoid(r["beta_market"] * logit(cur["p_market_a"]) + r["beta_model"] * logit(cur["p_a"]))))
    past = val[val["y"].notna()]
    if len(past) >= 200:
        r = encompassing_regression(past["y"], past["p_market_a"], past["p_a"])
        coefs["test"] = (r["beta_market"], r["beta_model"])
        te = base[base["split"] == "test"]
        out.append(te.assign(p_a=sigmoid(r["beta_market"] * logit(te["p_market_a"]) + r["beta_model"] * logit(te["p_a"]))))
    if not out:
        return pd.DataFrame(columns=preds_all.columns), coefs
    res = pd.concat(out, ignore_index=True).drop(columns=["p_market_a"]).assign(model="blend")
    return res, coefs


def run_backtests(cfg: dict, processed_dir: Path, artifacts_dir: Path) -> dict:
    processed_dir, artifacts_dir = Path(processed_dir), Path(artifacts_dir)
    out_dir = artifacts_dir / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_all = pd.read_parquet(artifacts_dir / "predictions.parquet")
    feats = pd.read_parquet(processed_dir / "features.parquet")
    fights = pd.read_parquet(processed_dir / "fights.parquet")
    meta = json.loads((artifacts_dir / "model_meta.json").read_text())
    odds_path = processed_dir / "odds.parquet"
    odds = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()
    mkt = market_probabilities(odds, fights, feats, cfg)

    result: dict = {"betting_model": meta["betting_model"], "reference_bookmaker": cfg["odds"]["reference_bookmaker"],
                    "params": cfg["backtest"], "decision_rule": {
                        "decision_hours_before_commence": cfg["odds"]["decision_hours_before_commence"],
                        "accept_date_only_snapshots": cfg["odds"]["accept_date_only_snapshots"]}}
    if mkt.empty:
        result["status"] = "no_odds"
        result["message"] = (
            "Aucune cote exploitable : il faut, pour chaque combat, les cotes décimales des DEUX combattants "
            "chez un même bookmaker au même instant, observées avant l'instant de décision "
            "(schéma : fight_id, fighter_id, bookmaker, snapshot_timestamp, commence_timestamp, decimal_odds). "
            "Voir docs/odds_import_format.md. Le module de prédiction reste utilisable.")
        (out_dir / "backtest_summary.json").write_text(json.dumps(result, indent=2, default=str))
        return result

    result["status"] = "ok"
    result["odds_snapshot_precision"] = mkt["snapshot_precision"].value_counts().to_dict()
    primary = meta["betting_model"]
    seed = cfg["project"]["random_seed"]
    n_boot = cfg["backtest"]["bootstrap"]["n_resamples"]
    blend, blend_coefs = blend_predictions(preds_all, mkt, primary)
    blend.to_parquet(artifacts_dir / "blend_predictions.parquet", index=False)
    result["blend_coefficients"] = {str(k): {"beta_market": v[0], "beta_model": v[1]} for k, v in blend_coefs.items()}
    preds_ext = pd.concat([preds_all, blend], ignore_index=True)
    result["primary"] = _run_variant(cfg, preds_ext, mkt, primary, out_dir, "primary", seed, n_boot, descriptive=True)
    result["blend"] = _run_variant(cfg, preds_ext, mkt, "blend", out_dir, "blend", seed, n_boot, descriptive=False)
    (out_dir / "backtest_summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _run_variant(cfg, preds_all, mkt, model, out_dir, tag, seed, n_boot, descriptive) -> dict:
    result = {"model": model}
    for split in ("validation", "test"):
        cands = build_candidates(preds_all, mkt, model, split)
        with_odds = cands.dropna(subset=["odds_a", "odds_b"])
        res_split = {"n_fights_predicted": int(len(cands)), "n_fights_with_odds": int(len(with_odds)),
                     "date_min": str(cands["event_date"].min().date()) if len(cands) else None,
                     "date_max": str(cands["event_date"].max().date()) if len(cands) else None,
                     "strategies": {}}
        if with_odds.empty:
            res_split["status"] = "no_odds_in_period"
            result[split] = res_split
            continue
        # evaluation restricted to events for which odds exist (other events: no bet possible)
        cands = cands[cands["event_id"].isin(with_odds["event_id"])]
        for strat in STRATEGIES:
            params = BacktestParams.from_config(cfg, strat)
            bets, events = run_backtest(cands, params)
            s = summarize(bets, events, params.initial_bankroll)
            s["by_year"] = by_period(bets).to_dict(orient="records")
            res_split["strategies"][strat] = s
            bets.to_parquet(out_dir / f"bets_{tag}_{split}_{strat}.parquet", index=False)
            events.to_parquet(out_dir / f"events_{tag}_{split}_{strat}.parquet", index=False)
        # sensitivity: EV threshold x odds haircut, descriptive only
        sens = []
        for thr in cfg["backtest"]["sensitivity"]["ev_thresholds"]:
            for h in cfg["backtest"]["sensitivity"]["odds_haircuts"]:
                for strat in STRATEGIES:
                    params = BacktestParams.from_config(cfg, strat, ev_threshold=thr, odds_haircut=h)
                    bets, events = run_backtest(cands, params)
                    s = summarize(bets, events, params.initial_bankroll)
                    sens.append({"ev_threshold": thr, "odds_haircut": h, "strategy": strat,
                                 **{k: s[k] for k in ("n_bets", "total_staked", "net_profit", "roi", "max_drawdown",
                                                      "final_bankroll")}})
        pd.DataFrame(sens).to_parquet(out_dir / f"sensitivity_{tag}_{split}.parquet", index=False)
        # draw settled as a loss instead of a refund
        alt = BacktestParams.from_config(cfg, "flat", draw_settlement="loss")
        b_alt, e_alt = run_backtest(cands, alt)
        res_split["flat_draw_as_loss"] = summarize(b_alt, e_alt, alt.initial_bankroll)
        res_split["bootstrap_unit_roi"] = bootstrap_unit_roi(cands, BacktestParams.from_config(cfg, "flat"),
                                                             n=n_boot, seed=seed)
        if not descriptive:
            result[split] = res_split
            continue
        # descriptive: same flat rule with the other models (not used for selection)
        other = {}
        for m in sorted(preds_all["model"].unique()):
            if m in ("baseline", "blend"):
                continue
            c = build_candidates(preds_all, mkt, m, split)
            c = c[c["event_id"].isin(with_odds["event_id"])]
            b, e = run_backtest(c, BacktestParams.from_config(cfg, "flat"))
            other[m] = {k: summarize(b, e, cfg["backtest"]["initial_bankroll"])[k]
                        for k in ("n_bets", "total_staked", "net_profit", "roi", "max_drawdown")}
        res_split["flat_by_model_descriptive"] = other
        result[split] = res_split
    return result
