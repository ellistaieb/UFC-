"""Stake caps, event exposure, settlement and no early reuse of capital."""

import numpy as np
import pandas as pd
import pytest

from ufc_quant.backtest.engine import BacktestParams, max_drawdown, run_backtest, settle


def _cands(rows):
    return pd.DataFrame(rows, columns=["fight_id", "event_id", "event_date", "p_a", "odds_a", "odds_b", "result_a"])


def test_per_fight_cap_and_event_exposure():
    # 8 very attractive bets in one event -> each capped at 1 %, total scaled to 5 %
    rows = [(f"f{i}", "E1", pd.Timestamp("2024-01-01"), 0.8, 2.0, 1.9, "win") for i in range(8)]
    p = BacktestParams(strategy="quarter_kelly", max_stake_fraction=0.01, max_event_exposure=0.05)
    bets, events = run_backtest(_cands(rows), p)
    assert bets["stake"].max() <= 0.01 * 1000 + 1e-9
    assert bets["stake"].sum() == pytest.approx(0.05 * 1000)


def test_stakes_use_bankroll_before_event_and_no_early_reuse():
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")
    rows = [("f1", "E1", d1, 0.6, 10.0, 1.1, "win"),   # huge win in event 1
            ("f2", "E1", d1, 0.6, 2.0, 1.9, "loss"),
            ("f3", "E2", d2, 0.6, 2.0, 1.9, "win")]
    p = BacktestParams(strategy="fixed_fraction", fraction=0.005, max_stake_fraction=0.01)
    bets, events = run_backtest(_cands(rows), p)
    e1 = bets[bets["event_id"] == "E1"]
    # both bets of event 1 sized on 1000, although f1 (a win) is listed first
    assert (e1["bankroll_before_event"] == 1000).all()
    np.testing.assert_allclose(e1["stake"], 5.0)
    # winnings of event 1 only available for event 2
    b1 = 1000 + 5 * 9 - 5
    e2 = bets[bets["event_id"] == "E2"].iloc[0]
    assert e2["bankroll_before_event"] == pytest.approx(b1)
    assert e2["stake"] == pytest.approx(0.005 * b1)


def test_event_stakes_never_exceed_available_bankroll():
    rows = [(f"f{i}", "E1", pd.Timestamp("2024-01-01"), 0.9, 3.0, 1.2, "loss") for i in range(50)]
    p = BacktestParams(strategy="flat", flat_units=500, max_stake_fraction=1.0, max_event_exposure=1.0)
    bets, events = run_backtest(_cands(rows), p)
    assert bets["stake"].sum() <= 1000 + 1e-9
    assert events["bankroll_after"].iloc[-1] >= 0


def test_settlement_rules():
    p = BacktestParams()
    assert settle("win", "A", 10, 2.5, p) == (15.0, "win")
    assert settle("loss", "A", 10, 2.5, p) == (-10, "loss")
    assert settle("loss", "B", 10, 1.8, p) == pytest.approx((8.0, "win"))
    assert settle("no_contest", "A", 10, 2.5, p) == (0.0, "refund")
    assert settle("draw", "B", 10, 2.5, p) == (0.0, "refund")
    p_loss = BacktestParams(draw_settlement="loss")
    assert settle("draw", "B", 10, 2.5, p_loss) == (-10, "loss")


def test_threshold_and_one_side_per_fight():
    rows = [("f1", "E1", pd.Timestamp("2024-01-01"), 0.50, 2.00, 2.00, "win"),   # EV 0 -> no bet
            ("f2", "E1", pd.Timestamp("2024-01-01"), 0.30, 2.00, 1.80, "loss")]  # B: 0.7*1.8-1 = 0.26
    bets, _ = run_backtest(_cands(rows), BacktestParams(ev_threshold=0.03))
    assert list(bets["fight_id"]) == ["f2"] and list(bets["side"]) == ["B"]


def test_odds_haircut_reduces_payout():
    rows = [("f1", "E1", pd.Timestamp("2024-01-01"), 0.7, 2.0, 1.9, "win")]
    b0, _ = run_backtest(_cands(rows), BacktestParams(strategy="flat"))
    b1, _ = run_backtest(_cands(rows), BacktestParams(strategy="flat", odds_haircut=0.05))
    assert b1["pnl"].iloc[0] < b0["pnl"].iloc[0]


def test_max_drawdown():
    assert max_drawdown([100, 120, 90, 130, 65]) == pytest.approx(0.5)
    assert max_drawdown([100, 110, 120]) == 0.0
