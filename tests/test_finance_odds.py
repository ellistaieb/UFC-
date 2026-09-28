"""Odds conversion, margin removal, Kelly, pre-decision odds selection, reconciliation."""

import numpy as np
import pandas as pd
import pytest

from ufc_quant.backtest.finance import (american_to_decimal, devig_proportional, expected_return, fair_odds,
                                        implied_probability, kelly_fraction, overround)
from ufc_quant.data.odds import pair_two_sided, select_pre_decision_odds, validate_standard_odds
from ufc_quant.data.reconcile import normalize_name, resolve_fighter_ids


def test_american_to_decimal():
    np.testing.assert_allclose(american_to_decimal([150, -200, 100]), [2.5, 1.5, 2.0])
    assert np.isnan(american_to_decimal([0])[0])


def test_implied_and_devig():
    oa, ob = 1.80, 2.10
    qa, qb = 1 / oa, 1 / ob
    assert implied_probability(oa) == pytest.approx(qa)
    assert overround(oa, ob) == pytest.approx(qa + qb - 1)
    pa, pb = devig_proportional(oa, ob)
    assert pa + pb == pytest.approx(1.0)
    assert pa == pytest.approx(qa / (qa + qb))


def test_ev_fair_odds_and_kelly():
    assert fair_odds(0.4) == pytest.approx(2.5)
    assert expected_return(0.5, 2.2) == pytest.approx(0.1)
    # Kelly (p*o - 1)/(o - 1)
    assert kelly_fraction(0.5, 2.2) == pytest.approx(0.1 / 1.2)
    # negative edge -> 0, fair bet -> 0
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)


def _odds_rows():
    commence = pd.Timestamp("2024-05-04 22:00", tz="UTC")
    rows = []
    for label, h, oa, ob in (("open", -48, 2.0, 1.9), ("t-6h", -6, 2.1, 1.8), ("close", -0.5, 2.5, 1.6),
                             ("in_play", 0.5, 5.0, 1.2)):
        ts = commence + pd.Timedelta(hours=h)
        rows += [dict(fight_id="F1", fighter_id="A", bookmaker="bk", snapshot_timestamp=ts,
                      commence_timestamp=commence, decimal_odds=oa),
                 dict(fight_id="F1", fighter_id="B", bookmaker="bk", snapshot_timestamp=ts,
                      commence_timestamp=commence, decimal_odds=ob)]
    return validate_standard_odds(pd.DataFrame(rows))


def test_only_odds_before_decision_time_are_used():
    fights = pd.DataFrame({"fight_id": ["F1"], "event_date": [pd.Timestamp("2024-05-04")]})
    sel = select_pre_decision_odds(_odds_rows(), fights, decision_hours_before=2, accept_imprecise=False)
    assert set(sel["decimal_odds"]) == {2.1, 1.8}  # t-6h, not close / in-play
    paired = pair_two_sided(sel, pd.DataFrame({"fight_id": ["F1"], "fighter_a_id": ["A"], "fighter_b_id": ["B"]}))
    assert paired["odds_a"].iloc[0] == 2.1 and paired["odds_b"].iloc[0] == 1.8


def test_missing_quote_is_not_filled_with_future_quote():
    odds = _odds_rows()
    # side B has only post-decision quotes -> pair must be absent, not built from a future quote
    odds = odds[~((odds["fighter_id"] == "B") & (odds["snapshot_timestamp"] < odds["commence_timestamp"] - pd.Timedelta(hours=2)))]
    fights = pd.DataFrame({"fight_id": ["F1"], "event_date": [pd.Timestamp("2024-05-04")]})
    sel = select_pre_decision_odds(odds, fights, 2, accept_imprecise=False)
    paired = pair_two_sided(sel, pd.DataFrame({"fight_id": ["F1"], "fighter_a_id": ["A"], "fighter_b_id": ["B"]}))
    assert paired.empty


def test_unknown_precision_requires_explicit_acceptance():
    odds = validate_standard_odds(pd.DataFrame({
        "fight_id": ["F1", "F1"], "fighter_id": ["A", "B"], "bookmaker": ["bk", "bk"],
        "snapshot_timestamp": [None, None], "decimal_odds": [1.9, 1.95]}))
    fights = pd.DataFrame({"fight_id": ["F1"], "event_date": [pd.Timestamp("2024-05-04")]})
    assert select_pre_decision_odds(odds, fights, 2, accept_imprecise=False).empty
    assert len(select_pre_decision_odds(odds, fights, 2, accept_imprecise=True)) == 2


def test_normalize_name():
    assert normalize_name("José Aldo Jr.") == "jose aldo"
    assert normalize_name("  Jan  Błachowicz ") == normalize_name("Jan Blachowicz")


def test_ambiguous_names_are_flagged():
    fighters = pd.DataFrame({"fighter_id": ["x1", "x2"], "name": ["Bruno Silva", "Bruno Silva"],
                             "weight_lbs": [185.0, 185.0], "dob": pd.to_datetime(["1989-01-01", "1990-01-01"])})
    parts = pd.DataFrame({"fight_id": ["f"], "event_date": [pd.Timestamp("2022-01-01")],
                          "weight_class": ["Middleweight Bout"], "name": ["Bruno Silva"]})
    out = resolve_fighter_ids(parts, fighters)
    assert out["id_status"].iloc[0] == "ambiguous"
    assert out["fighter_id"].iloc[0].startswith("ambig_")
    # disambiguation by weight class when possible
    fighters.loc[1, "weight_lbs"] = 125.0
    parts.loc[0, "weight_class"] = "Flyweight Bout"
    out = resolve_fighter_ids(parts, fighters)
    assert out["id_status"].iloc[0] == "disambiguated" and out["fighter_id"].iloc[0] == "x2"
