"""No future information in features; as-of path == training path; orientation independence."""

import numpy as np
import pandas as pd

from ufc_quant.features.engine import build_features, features_as_of


def _feature_cols(df):
    return [c for c in df.columns if c.startswith(("a_", "b_", "diff_"))]


def test_features_ignore_future_fights(synth, cfg):
    fights, stats, fighters = synth["fights"], synth["fight_stats"], synth["fighters"]
    full = build_features(fights, stats, fighters, cfg)
    cutoff = fights["event_date"].sort_values().iloc[len(fights) // 2]

    # Remove all later fights AND scramble results/stats of the cutoff date itself
    past = fights[fights["event_date"] <= cutoff].copy()
    same_day = past["event_date"] == cutoff
    past.loc[same_day, "result"] = past.loc[same_day, "result"].map(
        {"fighter_1": "fighter_2", "fighter_2": "fighter_1"}).fillna(past.loc[same_day, "result"])
    stats_mod = stats[stats["fight_id"].isin(past["fight_id"])].copy()
    stats_mod.loc[stats_mod["fight_id"].isin(past.loc[same_day, "fight_id"]), "sig_landed"] = 999
    trunc = build_features(past, stats_mod, fighters, cfg)

    cols = _feature_cols(full)
    a = full[full["event_date"] <= cutoff].set_index("fight_id").sort_index()[cols]
    b = trunc.set_index("fight_id").sort_index()[cols]
    pd.testing.assert_frame_equal(a, b)


def test_as_of_equals_training_row(synth, cfg):
    fights, stats, fighters = synth["fights"], synth["fight_stats"], synth["fighters"]
    full = build_features(fights, stats, fighters, cfg)
    row = full.iloc[len(full) * 3 // 4]
    f = fights.set_index("fight_id").loc[row["fight_id"]]
    q = features_as_of(fights, stats, fighters, cfg, row["fighter_a_id"], row["fighter_b_id"], row["event_date"],
                       scheduled_rounds=f["scheduled_rounds"], is_title=f["is_title"], is_women=f["is_women"],
                       weight_class=f["weight_class"])
    for c in _feature_cols(full):
        assert np.isclose(q[c], row[c], equal_nan=True), c


def test_orientation_independent_of_listing_order(synth, cfg):
    """UFC Stats lists the winner first; the canonical orientation must not depend on it."""
    fights, stats, fighters = synth["fights"], synth["fight_stats"], synth["fighters"]
    swapped = fights.copy()
    for c in ("id", "name", "id_status"):
        swapped[f"fighter_1_{c}"], swapped[f"fighter_2_{c}"] = fights[f"fighter_2_{c}"], fights[f"fighter_1_{c}"]
    swapped["result"] = fights["result"].map({"fighter_1": "fighter_2", "fighter_2": "fighter_1"}).fillna(fights["result"])
    a = build_features(fights, stats, fighters, cfg).set_index("fight_id").sort_index()
    b = build_features(swapped, stats, fighters, cfg).set_index("fight_id").sort_index()
    cols = _feature_cols(a) + ["y", "fighter_a_id"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_debutant_has_explicit_missing_layoff(synth, cfg):
    feats = build_features(synth["fights"], synth["fight_stats"], synth["fighters"], cfg)
    debut = feats["a_n_fights"] == 0
    assert debut.any()
    assert feats.loc[debut, "a_days_since_last"].isna().all()
    # shrunk rates of a debutant equal the population prior (finite, not 0)
    later = debut & (feats["event_date"] > feats["event_date"].min())
    assert (feats.loc[later, "a_sig_landed_pm"] > 0).all()
