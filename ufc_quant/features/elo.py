"""Chronological Elo rating.

Rules
- All fights on the same date use the ratings from before that date (no same-day leakage),
  updates are applied after the whole date batch.
- Win = 1, loss = 0, draw = `draw_score` (0.5), no contest = no update.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(r_b, dtype=float) - np.asarray(r_a, dtype=float)) / 400.0))


class EloRating:
    def __init__(self, k: float = 32.0, initial: float = 1500.0, draw_score: float = 0.5):
        self.k = k
        self.initial = initial
        self.draw_score = draw_score
        self.ratings: dict[str, float] = defaultdict(lambda: initial)

    def get(self, fighter_id: str) -> float:
        return self.ratings[fighter_id] if fighter_id in self.ratings else self.initial

    def score_a(self, result_a: str) -> float | None:
        return {"win": 1.0, "loss": 0.0, "draw": self.draw_score}.get(result_a)

    def update_batch(self, games: list[tuple[str, str, str]]) -> None:
        """games: (a_id, b_id, result_a in {'win','loss','draw','no_contest'}) sharing one date."""
        deltas: dict[str, float] = defaultdict(float)
        for a, b, res in games:
            s = self.score_a(res)
            if s is None:
                continue
            e = float(elo_expected(self.get(a), self.get(b)))
            deltas[a] += self.k * (s - e)
            deltas[b] -= self.k * (s - e)
        for fid, d in deltas.items():
            self.ratings[fid] = self.get(fid) + d


def result_for_a(result: str, a_is_fighter_1: bool) -> str:
    if result == "fighter_1":
        return "win" if a_is_fighter_1 else "loss"
    if result == "fighter_2":
        return "loss" if a_is_fighter_1 else "win"
    return result if result in ("draw", "no_contest") else "no_contest"


def elo_pre_fight(fights: pd.DataFrame, k: float, initial: float = 1500.0, draw_score: float = 0.5) -> pd.DataFrame:
    """Pre-fight Elo of fighter_1 and fighter_2 for every fight (chronological)."""
    elo = EloRating(k, initial, draw_score)
    rows = []
    for _, day in fights.sort_values("event_date").groupby("event_date", sort=True):
        games = []
        for r in day.itertuples(index=False):
            rows.append((r.fight_id, elo.get(r.fighter_1_id), elo.get(r.fighter_2_id)))
            games.append((r.fighter_1_id, r.fighter_2_id, result_for_a(r.result, True)))
        elo.update_batch(games)
    return pd.DataFrame(rows, columns=["fight_id", "elo_1_pre", "elo_2_pre"])


def tune_elo_k(fights: pd.DataFrame, k_grid: list[float], eval_start: str, eval_end: str,
               initial: float = 1500.0, draw_score: float = 0.5) -> tuple[float, dict]:
    """Pick K minimising log-loss of the Elo probability on decided fights in [eval_start, eval_end)."""
    scores = {}
    mask = (fights["event_date"] >= eval_start) & (fights["event_date"] < eval_end) & \
        fights["result"].isin(["fighter_1", "fighter_2"])
    for k in k_grid:
        pre = elo_pre_fight(fights, k, initial, draw_score).set_index("fight_id")
        sub = fights[mask]
        p1 = elo_expected(pre.loc[sub["fight_id"], "elo_1_pre"].values, pre.loc[sub["fight_id"], "elo_2_pre"].values)
        y = (sub["result"] == "fighter_1").values.astype(float)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        scores[float(k)] = float(-np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
    best = min(scores, key=scores.get)
    return best, scores
