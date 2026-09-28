"""Shared fixtures: a small SYNTHETIC dataset built through the real transform code."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from ufc_quant.config import Paths, load_config
from ufc_quant.data.synthetic import build_synthetic_odds, write_synthetic_raw
from ufc_quant.data.transform_ufcstats import transform_ufcstats


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c = copy.deepcopy(c)
    c["synthetic"].update(n_fighters=120, start_date="2019-01-05", end_date="2021-12-25",
                          days_between_events=21, fights_per_event=6)
    c["splits"].update(train_start="2019-01-01", walk_forward_years=[2020], test_start="2021-01-01")
    c["odds"]["reference_bookmaker"] = "SYNTHETIC_BOOK"
    return c


@pytest.fixture(scope="session")
def synth(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("synthetic")
    paths = Paths(root / "raw", root / "processed", root / "artifacts", root / "reports", "synthetic").ensure()
    write_synthetic_raw(cfg, paths)
    transform_ufcstats(paths.raw, paths.processed)
    build_synthetic_odds(paths)
    data = {n: pd.read_parquet(paths.processed / f"{n}.parquet") for n in ("fights", "fight_stats", "fighters", "odds")}
    data["paths"] = paths
    return data
