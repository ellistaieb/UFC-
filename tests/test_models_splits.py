"""Temporal separation of events and A/B symmetry of the models."""

import numpy as np
import pandas as pd
import pytest

from ufc_quant.evaluation.splits import assert_event_disjoint, final_split, walk_forward_folds
from ufc_quant.features.engine import build_features
from ufc_quant.models.core import TemperatureCalibrator, augment, make_model, swap_orientation


@pytest.fixture(scope="module")
def feats(synth, cfg):
    return build_features(synth["fights"], synth["fight_stats"], synth["fighters"], cfg)


def test_walk_forward_is_chronological_and_event_disjoint(feats, cfg):
    folds = list(walk_forward_folds(feats, cfg, include_undecided=True))
    assert folds
    for _, tr, va in folds:
        assert_event_disjoint(tr, va)
        assert tr["event_date"].max() < va["event_date"].min()
        assert va["event_date"].max() < pd.Timestamp(cfg["splits"]["test_start"])


def test_final_test_is_after_training(feats, cfg):
    tr, te = final_split(feats, cfg)
    assert_event_disjoint(tr, te)
    assert te["event_date"].min() >= pd.Timestamp(cfg["splits"]["test_start"])
    # every event is entirely in one segment
    seg = pd.concat([tr.assign(s="tr"), te.assign(s="te")]).groupby("event_id")["s"].nunique()
    assert (seg == 1).all()


def test_assert_event_disjoint_detects_overlap(feats):
    with pytest.raises(AssertionError):
        assert_event_disjoint(feats.iloc[:50], feats.iloc[40:90])


def test_swap_is_an_involution(feats):
    back = swap_orientation(swap_orientation(feats))
    pd.testing.assert_frame_equal(back, feats)


def test_augmented_pairs_stay_together(feats):
    aug = augment(feats)
    assert len(aug) == 2 * len(feats)
    assert (aug.groupby("fight_id")["event_date"].nunique() == 1).all()


@pytest.mark.parametrize("family,params", [
    ("elo", {}), ("logistic", {"C": 0.1}),
    ("gbm", {"learning_rate": 0.1, "max_depth": 2, "max_iter": 30, "min_samples_leaf": 20}),
])
def test_model_symmetry(feats, cfg, family, params):
    tr, te = final_split(feats, cfg)
    m = make_model(family, params).fit(tr)
    p_ab = m.predict_proba_a(te)
    p_ba = m.predict_proba_a(swap_orientation(te))
    np.testing.assert_allclose(p_ab + p_ba, 1.0, atol=1e-9)


def test_temperature_scaling_preserves_symmetry():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 500)
    y = (rng.random(500) < p).astype(float)
    cal = TemperatureCalibrator().fit(p, y)
    np.testing.assert_allclose(cal.transform(p) + cal.transform(1 - p), 1.0, atol=1e-9)
