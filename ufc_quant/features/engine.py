"""Chronological, leakage-free feature engine.

One single code path produces both the training features and the "as of date" features
used by the dashboard:

    for each date (all events of that date form one batch):
        1. snapshot the state of every participant  -> pre-fight features
        2. only then update fighter states, Elo and population priors with the results

A fighter state therefore only ever contains fights strictly before the current date.
Career statistics scraped today are never used; only static attributes (date of birth,
height, reach) come from the current fighter profile.

Handling of special outcomes
- draw: counts as a fight (experience, time, statistics), scores 0.5 in form/win-rate and Elo.
- no contest: counts for experience, time and statistics, not for results nor Elo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ufc_quant.data.reconcile import weight_class_limit
from ufc_quant.features.elo import EloRating, result_for_a

# Statistics accumulated from fight_stats (own and opponent side)
OWN_STATS = ["sig_landed", "sig_att", "td_landed", "td_att", "sub_att", "kd", "ctrl_seconds"]

# Per-fighter features produced by FighterState.snapshot (all antisymmetric in the matchup)
FIGHTER_FEATURES = [
    "age_years", "height_cm", "reach_cm", "n_fights", "win_rate_shrunk", "recent_form",
    "days_since_last", "sig_landed_pm", "sig_absorbed_pm", "sig_acc", "sig_def",
    "td_per15", "td_acc", "td_def", "sub_att_per15", "ctrl_share", "elo", "opp_elo_mean",
]
CONTEXT_FEATURES = ["five_rounds", "is_title", "is_women", "weight_limit_lbs"]


@dataclass
class FighterState:
    n_fights: int = 0
    wins: float = 0.0
    losses: float = 0.0
    draws: float = 0.0
    last_date: pd.Timestamp | None = None
    recent_scores: list = field(default_factory=list)
    stats_seconds: float = 0.0
    n_fights_with_stats: int = 0
    own: dict = field(default_factory=lambda: {k: 0.0 for k in OWN_STATS})
    opp: dict = field(default_factory=lambda: {k: 0.0 for k in OWN_STATS})
    opp_elo_sum: float = 0.0
    n_opp_elo: int = 0


class Priors:
    """Population totals over all fighter-fights strictly before the current date."""

    def __init__(self):
        self.tot = {k: 0.0 for k in OWN_STATS}
        self.seconds = 0.0

    def add(self, own: dict, seconds: float) -> None:
        for k in OWN_STATS:
            self.tot[k] += own[k]
        self.seconds += seconds

    def rates(self) -> dict:
        minutes = self.seconds / 60.0
        t = self.tot
        nan = float("nan")
        return {
            "sig_pm": t["sig_landed"] / minutes if minutes > 0 else nan,
            "sig_acc": t["sig_landed"] / t["sig_att"] if t["sig_att"] > 0 else nan,
            "td_pm": t["td_landed"] / minutes if minutes > 0 else nan,
            "td_acc": t["td_landed"] / t["td_att"] if t["td_att"] > 0 else nan,
            "sub_pm": t["sub_att"] / minutes if minutes > 0 else nan,
            "ctrl_share": t["ctrl_seconds"] / self.seconds if self.seconds > 0 else nan,
        }


class FeatureEngine:
    def __init__(self, fights: pd.DataFrame, fight_stats: pd.DataFrame, fighters: pd.DataFrame, cfg: dict,
                 elo_k: float | None = None):
        fcfg = cfg["features"]
        self.k_min = float(fcfg["shrink_minutes"])
        self.k_att = float(fcfg["shrink_attempts"])
        self.form_window = int(fcfg["recent_form_window"])
        ecfg = fcfg["elo"]
        self.elo = EloRating(elo_k if elo_k is not None else ecfg["k_factor"], ecfg["initial_rating"], ecfg["draw_score"])
        self.fights = fights.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
        self.stats = {(r.fight_id, r.fighter_id): r for r in fight_stats.itertuples(index=False)}
        f = fighters.drop_duplicates("fighter_id").set_index("fighter_id")
        self.static = {
            "dob": f["dob"].to_dict(), "height_cm": f["height_cm"].to_dict(), "reach_cm": f["reach_cm"].to_dict(),
        }
        self.states: dict[str, FighterState] = {}
        self.priors = Priors()
        self.current_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------ snapshot
    def state(self, fid: str) -> FighterState:
        return self.states.get(fid) or FighterState()

    def snapshot(self, fid: str, date: pd.Timestamp) -> dict:
        s = self.state(fid)
        pr = self.priors.rates()
        k, m = self.k_min, self.k_att
        minutes = s.stats_seconds / 60.0
        dob = self.static["dob"].get(fid)
        decided = s.wins + s.losses + s.draws
        recent = s.recent_scores[-self.form_window:]

        def rate(x, prior):  # shrunk per-minute rate
            return (x + k * prior) / (minutes + k)

        def ratio(num, den, prior):  # shrunk proportion
            return (num + m * prior) / (den + m)

        return {
            "age_years": (date - dob).days / 365.25 if dob is not None and pd.notna(dob) else np.nan,
            "height_cm": self.static["height_cm"].get(fid, np.nan),
            "reach_cm": self.static["reach_cm"].get(fid, np.nan),
            "n_fights": float(s.n_fights),
            "win_rate_shrunk": (s.wins + 0.5 * s.draws + 1.0) / (decided + 2.0),
            "recent_form": (sum(recent) + 1.0) / (len(recent) + 2.0),
            "days_since_last": float((date - s.last_date).days) if s.last_date is not None else np.nan,
            "sig_landed_pm": rate(s.own["sig_landed"], pr["sig_pm"]),
            "sig_absorbed_pm": rate(s.opp["sig_landed"], pr["sig_pm"]),
            "sig_acc": ratio(s.own["sig_landed"], s.own["sig_att"], pr["sig_acc"]),
            "sig_def": 1.0 - ratio(s.opp["sig_landed"], s.opp["sig_att"], pr["sig_acc"]),
            "td_per15": 15.0 * rate(s.own["td_landed"], pr["td_pm"]),
            "td_acc": ratio(s.own["td_landed"], s.own["td_att"], pr["td_acc"]),
            "td_def": 1.0 - ratio(s.opp["td_landed"], s.opp["td_att"], pr["td_acc"]),
            "sub_att_per15": 15.0 * rate(s.own["sub_att"], pr["sub_pm"]),
            "ctrl_share": (s.own["ctrl_seconds"] + k * 60 * pr["ctrl_share"]) / (s.stats_seconds + k * 60),
            "elo": self.elo.get(fid),
            "opp_elo_mean": (s.opp_elo_sum + self.elo.initial) / (s.n_opp_elo + 1.0),
            "n_fights_with_stats": float(s.n_fights_with_stats),
        }

    # ------------------------------------------------------------------ update
    def _update_batch(self, day: pd.DataFrame, date: pd.Timestamp) -> None:
        games, pending = [], []
        pre_elo = {}
        for r in day.itertuples(index=False):
            for fid in (r.fighter_1_id, r.fighter_2_id):
                pre_elo[fid] = self.elo.get(fid)
        for r in day.itertuples(index=False):
            res1 = result_for_a(r.result, True)
            games.append((r.fighter_1_id, r.fighter_2_id, res1))
            s1 = self.stats.get((r.fight_id, r.fighter_1_id))
            s2 = self.stats.get((r.fight_id, r.fighter_2_id))
            usable = bool(r.has_stats) and s1 is not None and s2 is not None and pd.notna(r.fight_seconds) \
                and all(pd.notna(getattr(x, c)) for x in (s1, s2) for c in OWN_STATS)
            for fid, oid, own, opp, res in ((r.fighter_1_id, r.fighter_2_id, s1, s2, res1),
                                            (r.fighter_2_id, r.fighter_1_id, s2, s1, _flip(res1))):
                pending.append((fid, oid, res, own if usable else None, opp if usable else None,
                                r.fight_seconds if usable else None))
        for fid, oid, res, own, opp, secs in pending:
            s = self.states.setdefault(fid, FighterState())
            s.n_fights += 1
            s.last_date = date
            if res in ("win", "loss", "draw"):
                score = {"win": 1.0, "loss": 0.0, "draw": 0.5}[res]
                s.wins += res == "win"
                s.losses += res == "loss"
                s.draws += res == "draw"
                s.recent_scores.append(score)
            s.opp_elo_sum += pre_elo[oid]
            s.n_opp_elo += 1
            if own is not None:
                s.n_fights_with_stats += 1
                s.stats_seconds += float(secs)
                own_d = {c: float(getattr(own, c)) for c in OWN_STATS}
                for c in OWN_STATS:
                    s.own[c] += own_d[c]
                    s.opp[c] += float(getattr(opp, c))
        # population priors after the whole batch (same-date information stays hidden)
        for fid, oid, res, own, opp, secs in pending:
            if own is not None:
                self.priors.add({c: float(getattr(own, c)) for c in OWN_STATS}, float(secs))
        self.elo.update_batch(games)

    # ------------------------------------------------------------------ public API
    def run(self, until: pd.Timestamp | None = None, collect: bool = True) -> pd.DataFrame | None:
        """Process all dates < until (all dates if None). Returns fight-level features if collect."""
        rows = []
        fights = self.fights if until is None else self.fights[self.fights["event_date"] < until]
        for date, day in fights.groupby("event_date", sort=True):
            self.current_date = date
            if collect:
                for r in day.itertuples(index=False):
                    rows.append(self._fight_row(r, date))
            self._update_batch(day, date)
        return pd.DataFrame(rows) if collect else None

    def _fight_row(self, r, date) -> dict:
        # canonical orientation independent of the result and of UFC Stats' listing order
        a_is_1 = str(r.fighter_1_id) <= str(r.fighter_2_id)
        a, b = (r.fighter_1_id, r.fighter_2_id) if a_is_1 else (r.fighter_2_id, r.fighter_1_id)
        res_a = result_for_a(r.result, a_is_1)
        row = {
            "fight_id": r.fight_id, "event_id": r.event_id, "event_date": date,
            "fighter_a_id": a, "fighter_b_id": b, "result_a": res_a,
            "y": {"win": 1.0, "loss": 0.0}.get(res_a, np.nan),
            "id_ambiguous": bool(r.id_ambiguous), "has_stats": bool(r.has_stats),
        }
        row.update(context_features(r.scheduled_rounds, r.is_title, r.is_women, r.weight_class))
        row.update(matchup_features(self.snapshot(a, date), self.snapshot(b, date)))
        return row


def _flip(res: str) -> str:
    return {"win": "loss", "loss": "win"}.get(res, res)


def context_features(scheduled_rounds, is_title, is_women, weight_class) -> dict:
    return {
        "five_rounds": float(scheduled_rounds == 5) if pd.notna(scheduled_rounds) else np.nan,
        "is_title": float(bool(is_title)),
        "is_women": float(bool(is_women)),
        "weight_limit_lbs": weight_class_limit(weight_class),
    }


def matchup_features(sa: dict, sb: dict) -> dict:
    out = {}
    for k in FIGHTER_FEATURES + ["n_fights_with_stats"]:
        out[f"a_{k}"] = sa[k]
        out[f"b_{k}"] = sb[k]
    for k in FIGHTER_FEATURES:
        out[f"diff_{k}"] = sa[k] - sb[k]
    return out


DIFF_FEATURES = [f"diff_{k}" for k in FIGHTER_FEATURES]


def build_features(fights, fight_stats, fighters, cfg, elo_k=None) -> pd.DataFrame:
    return FeatureEngine(fights, fight_stats, fighters, cfg, elo_k).run()


def features_as_of(fights, fight_stats, fighters, cfg, a_id: str, b_id: str, date, elo_k=None,
                   scheduled_rounds=3, is_title=False, is_women=False, weight_class="") -> dict:
    """Features of a (possibly hypothetical) matchup using only fights strictly before `date`."""
    date = pd.Timestamp(date)
    eng = FeatureEngine(fights, fight_stats, fighters, cfg, elo_k)
    eng.run(until=date, collect=False)
    row = {"fighter_a_id": a_id, "fighter_b_id": b_id, "event_date": date}
    row.update(context_features(scheduled_rounds, is_title, is_women, weight_class))
    row.update(matchup_features(eng.snapshot(a_id, date), eng.snapshot(b_id, date)))
    row["a_n_prior"] = eng.state(a_id).n_fights
    row["b_n_prior"] = eng.state(b_id).n_fights
    return row
