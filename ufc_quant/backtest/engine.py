"""Event-driven betting simulation with risk constraints (fictitious bankroll, no real bets).

Timeline for each event (events processed in chronological order):
    1. B0 = bankroll available BEFORE the event (all previous events settled).
    2. For every fight: choose at most one side, the one with the highest estimated expected
       return p * o - 1, if it exceeds the threshold. Stake computed from B0 only.
    3. Per-fight cap (max_stake_fraction * B0) then per-event cap (max_event_exposure * B0,
       proportional scaling). Funds are reserved: total stakes never exceed B0.
    4. All bets of the event are settled together; gains become available only for the
       NEXT event (no reuse of winnings whose outcome would not yet be known).

Settlement rules (configurable, see config.backtest.settlement)
    win -> + stake * (o - 1); loss -> - stake;
    no contest -> refund (0); draw -> refund (0) by default, or loss.
    Cancelled fights are absent from the data (they would be refunded): not simulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ufc_quant.backtest.finance import kelly_fraction


@dataclass
class BacktestParams:
    initial_bankroll: float = 1000.0
    ev_threshold: float = 0.03
    max_stake_fraction: float = 0.01
    max_event_exposure: float = 0.05
    strategy: str = "flat"            # flat | fixed_fraction | quarter_kelly
    flat_units: float = 10.0
    fraction: float = 0.005
    kelly_multiplier: float = 0.25
    odds_haircut: float = 0.0         # net odds degraded to (o - 1) * (1 - haircut)
    draw_settlement: str = "refund"   # refund | loss
    nc_settlement: str = "refund"

    @classmethod
    def from_config(cls, cfg: dict, strategy: str, **overrides) -> "BacktestParams":
        b = cfg["backtest"]
        s = b["strategies"]
        p = cls(initial_bankroll=b["initial_bankroll"], ev_threshold=b["ev_threshold"],
                max_stake_fraction=b["max_stake_fraction"], max_event_exposure=b["max_event_exposure"],
                strategy=strategy, flat_units=s["flat"]["units"], fraction=s["fixed_fraction"]["fraction"],
                kelly_multiplier=s["quarter_kelly"]["kelly_multiplier"],
                draw_settlement=b["settlement"]["draw"], nc_settlement=b["settlement"]["no_contest"])
        for k, v in overrides.items():
            setattr(p, k, v)
        return p


REQUIRED = ["fight_id", "event_id", "event_date", "p_a", "odds_a", "odds_b", "result_a"]


def effective_odds(o, haircut: float):
    return 1.0 + (np.asarray(o, dtype=float) - 1.0) * (1.0 - haircut)


def choose_bets(cands: pd.DataFrame, params: BacktestParams) -> pd.DataFrame:
    """Side selection and estimated expected return, independent of bankroll."""
    c = cands.copy()
    c["o_a"] = effective_odds(c["odds_a"], params.odds_haircut)
    c["o_b"] = effective_odds(c["odds_b"], params.odds_haircut)
    ev_a = c["p_a"] * c["o_a"] - 1.0
    ev_b = (1.0 - c["p_a"]) * c["o_b"] - 1.0
    pick_a = ev_a >= ev_b
    c["side"] = np.where(pick_a, "A", "B")
    c["p_side"] = np.where(pick_a, c["p_a"], 1.0 - c["p_a"])
    c["odds_side"] = np.where(pick_a, c["o_a"], c["o_b"])
    c["ev"] = np.where(pick_a, ev_a, ev_b)
    c["kelly_full"] = kelly_fraction(c["p_side"], c["odds_side"])
    return c[c["ev"] > params.ev_threshold]


def settle(result_a: str, side: str, stake: float, odds: float, params: BacktestParams) -> tuple[float, str]:
    if result_a == "no_contest":
        return (0.0, "refund") if params.nc_settlement == "refund" else (-stake, "loss")
    if result_a == "draw":
        return (0.0, "refund") if params.draw_settlement == "refund" else (-stake, "loss")
    won = (result_a == "win") == (side == "A")
    return (stake * (odds - 1.0), "win") if won else (-stake, "loss")


def run_backtest(cands: pd.DataFrame, params: BacktestParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (bets, events) tables. `cands`: one row per fight with REQUIRED columns."""
    missing = [c for c in REQUIRED if c not in cands.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    chosen = choose_bets(cands.dropna(subset=["p_a", "odds_a", "odds_b"]), params)
    events = cands[["event_id", "event_date"]].drop_duplicates("event_id").sort_values(["event_date", "event_id"])
    by_event = {k: g for k, g in chosen.groupby("event_id")}
    bankroll = params.initial_bankroll
    bet_rows, ev_rows = [], []
    for ev in events.itertuples(index=False):
        b0 = bankroll
        g = by_event.get(ev.event_id)
        staked = pnl_total = 0.0
        n = 0
        if g is not None and b0 > 0:
            if params.strategy == "flat":
                raw = np.full(len(g), params.flat_units)
            elif params.strategy == "fixed_fraction":
                raw = np.full(len(g), params.fraction * b0)
            elif params.strategy == "quarter_kelly":
                raw = params.kelly_multiplier * g["kelly_full"].values * b0
            else:
                raise ValueError(params.strategy)
            stakes = np.minimum(raw, params.max_stake_fraction * b0)
            cap = min(params.max_event_exposure * b0, b0)
            if stakes.sum() > cap:
                stakes = stakes * cap / stakes.sum()
            for (r, stake) in zip(g.itertuples(index=False), stakes):
                if stake <= 0:
                    continue
                pnl, outcome = settle(r.result_a, r.side, float(stake), float(r.odds_side), params)
                bet_rows.append({"event_id": ev.event_id, "event_date": ev.event_date, "fight_id": r.fight_id,
                                 "side": r.side, "p_side": r.p_side, "odds": r.odds_side, "ev": r.ev,
                                 "kelly_full": r.kelly_full, "bankroll_before_event": b0, "stake": float(stake),
                                 "outcome": outcome, "pnl": pnl})
                staked += stake
                pnl_total += pnl
                n += 1
        bankroll = b0 + pnl_total
        ev_rows.append({"event_id": ev.event_id, "event_date": ev.event_date, "bankroll_before": b0,
                        "n_bets": n, "staked": staked, "pnl": pnl_total, "bankroll_after": bankroll})
    return pd.DataFrame(bet_rows), pd.DataFrame(ev_rows)


def max_drawdown(bankroll_path) -> float:
    """Largest peak-to-trough decline, as a fraction of the running peak."""
    x = np.asarray(bankroll_path, dtype=float)
    if len(x) == 0:
        return 0.0
    peak = np.maximum.accumulate(x)
    return float(np.max((peak - x) / peak))


def summarize(bets: pd.DataFrame, events: pd.DataFrame, initial: float) -> dict:
    staked = float(bets["stake"].sum()) if len(bets) else 0.0
    profit = float(bets["pnl"].sum()) if len(bets) else 0.0
    path = np.concatenate([[initial], events["bankroll_after"].values]) if len(events) else np.array([initial])
    return {
        "n_bets": int(len(bets)), "n_events_with_bets": int((events["n_bets"] > 0).sum()) if len(events) else 0,
        "total_staked": staked, "net_profit": profit,
        "roi": profit / staked if staked > 0 else float("nan"),
        "final_bankroll": float(path[-1]), "max_drawdown": max_drawdown(path),
        "hit_rate": float((bets["outcome"] == "win").mean()) if len(bets) else float("nan"),
        "mean_odds": float(bets["odds"].mean()) if len(bets) else float("nan"),
        "mean_estimated_ev": float(np.average(bets["ev"], weights=bets["stake"])) if len(bets) else float("nan"),
        "n_refunds": int((bets["outcome"] == "refund").sum()) if len(bets) else 0,
    }


def by_period(bets: pd.DataFrame, freq: str = "Y") -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(columns=["period", "n_bets", "staked", "profit", "roi"])
    b = bets.assign(period=pd.to_datetime(bets["event_date"]).dt.year if freq == "Y"
                    else pd.to_datetime(bets["event_date"]).dt.to_period(freq).astype(str))
    g = b.groupby("period").agg(n_bets=("pnl", "size"), staked=("stake", "sum"), profit=("pnl", "sum")).reset_index()
    g["roi"] = g["profit"] / g["staked"]
    return g


def bootstrap_unit_roi(cands: pd.DataFrame, params: BacktestParams, n: int = 2000, seed: int = 0) -> dict:
    """Event-level bootstrap of the ROI of 1-unit bets on the selected sides.

    Unit stakes make the ROI path-independent, so events can be resampled. Limits: assumes
    events are exchangeable (no regime change), ignores bankroll dynamics and parameter
    uncertainty; it is NOT used to pick a strategy.
    """
    chosen = choose_bets(cands.dropna(subset=["p_a", "odds_a", "odds_b"]), params)
    if chosen.empty:
        return {"n_bets": 0}
    pnl = [settle(r.result_a, r.side, 1.0, r.odds_side, params)[0] for r in chosen.itertuples(index=False)]
    d = pd.DataFrame({"event_id": chosen["event_id"].values, "pnl": pnl})
    g = d.groupby("event_id")["pnl"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    s, z = g["sum"].values, g["size"].values
    boot = s[idx].sum(axis=1) / z[idx].sum(axis=1)
    return {"n_bets": int(z.sum()), "n_events": int(len(g)), "roi_unit": float(s.sum() / z.sum()),
            "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
            "p_roi_le_0": float(np.mean(boot <= 0))}
