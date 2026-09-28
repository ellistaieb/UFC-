"""Command line: python -m ufc_quant <step> [--synthetic] [--refresh]

Steps: download, prepare, features, train, evaluate, backtest, report, all
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from ufc_quant.config import get_paths, load_config
from ufc_quant.pipeline import dataset_meta, step_features

log = logging.getLogger("ufc_quant")


def step_download(cfg, paths, refresh):
    if paths.dataset_kind == "synthetic":
        from ufc_quant.data.synthetic import write_synthetic_raw
        return write_synthetic_raw(cfg, paths)
    from ufc_quant.data.ingest_ufcstats import acquire_ufcstats
    from ufc_quant.data.odds import acquire_public_odds
    m1 = acquire_ufcstats(cfg, paths.raw, refresh=refresh)
    m2 = acquire_public_odds(cfg, paths.raw, refresh=refresh)
    return {"ufcstats": m1["source"], "odds": list(m2)}


def step_prepare(cfg, paths):
    from ufc_quant.data.odds import build_odds_table
    from ufc_quant.data.transform_ufcstats import transform_ufcstats
    q = transform_ufcstats(paths.raw, paths.processed)
    if paths.dataset_kind == "synthetic":
        from ufc_quant.data.synthetic import build_synthetic_odds
        o = build_synthetic_odds(paths)
    else:
        o = build_odds_table(cfg, paths.raw, paths.processed)
    meta = {"dataset_kind": paths.dataset_kind}
    (paths.processed / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    return {"ufcstats": q, "odds": o}


def step_train(cfg, paths):
    from ufc_quant.models.train import train_all
    feats = pd.read_parquet(paths.processed / "features.parquet")
    return train_all(feats, cfg, paths.artifacts, dataset_meta(paths))


def step_evaluate(cfg, paths):
    from ufc_quant.evaluation.evaluate import evaluate
    return evaluate(cfg, paths.processed, paths.artifacts)


def step_backtest(cfg, paths):
    from ufc_quant.backtest.run import run_backtests
    return run_backtests(cfg, paths.processed, paths.artifacts)


def step_report(cfg, paths):
    from ufc_quant.reporting.report import write_report
    return {"report": str(write_report(cfg, paths))}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ufc_quant")
    ap.add_argument("step", choices=["download", "prepare", "features", "train", "evaluate", "backtest", "report", "all"])
    ap.add_argument("--synthetic", action="store_true", help="démonstration sur données SYNTHÉTIQUES")
    ap.add_argument("--refresh", action="store_true", help="re-télécharger les sources (ETag)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    paths = get_paths(cfg, synthetic=args.synthetic).ensure()
    if args.synthetic:
        cfg["odds"]["reference_bookmaker"] = cfg["synthetic"]["reference_bookmaker"]
        log.warning("MODE DÉMONSTRATION : données SYNTHÉTIQUES, aucun résultat n'est une mesure réelle.")
    steps = ["download", "prepare", "features", "train", "evaluate", "backtest", "report"] if args.step == "all" else [args.step]
    for s in steps:
        log.info("=== %s (%s) ===", s, paths.dataset_kind)
        if s == "download":
            r = step_download(cfg, paths, args.refresh)
        elif s == "prepare":
            r = step_prepare(cfg, paths)
        elif s == "features":
            r = step_features(cfg, paths)
        elif s == "train":
            r = step_train(cfg, paths)
        elif s == "evaluate":
            r = step_evaluate(cfg, paths)
        elif s == "backtest":
            r = step_backtest(cfg, paths)
        else:
            r = step_report(cfg, paths)
        log.info("%s done: %s", s, json.dumps(r, default=str)[:600])


if __name__ == "__main__":
    main()
