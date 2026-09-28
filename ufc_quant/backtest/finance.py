"""Odds arithmetic for two-outcome markets (decimal odds)."""

from __future__ import annotations

import numpy as np


def american_to_decimal(american):
    """+150 -> 2.50 ; -200 -> 1.50. Zero / NaN -> NaN."""
    a = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dec = np.where(a > 0, 1 + a / 100.0, np.where(a < 0, 1 + 100.0 / np.abs(a), np.nan))
    return dec


def implied_probability(decimal_odds):
    """Raw implied probability q = 1 / o (includes the bookmaker margin)."""
    o = np.asarray(decimal_odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o > 1.0, 1.0 / o, np.nan)


def overround(odds_a, odds_b):
    """Bookmaker margin: q_a + q_b - 1 (same bookmaker, same snapshot)."""
    return implied_probability(odds_a) + implied_probability(odds_b) - 1.0


def devig_proportional(odds_a, odds_b):
    """Proportional margin removal p_i = q_i / (q_a + q_b).

    This is an approximation: it assumes the margin is spread proportionally to the raw
    implied probabilities, which ignores the favourite-longshot bias.
    Returns (p_a, p_b).
    """
    qa, qb = implied_probability(odds_a), implied_probability(odds_b)
    s = qa + qb
    return qa / s, qb / s


def fair_odds(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(p > 0, 1.0 / p, np.inf)


def expected_return(p, decimal_odds):
    """Expected profit per unit staked at the offered odds: p * o - 1."""
    return np.asarray(p, dtype=float) * np.asarray(decimal_odds, dtype=float) - 1.0


def kelly_fraction(p, decimal_odds):
    """Full Kelly fraction (p*o - 1) / (o - 1), floored at 0."""
    p = np.asarray(p, dtype=float)
    o = np.asarray(decimal_odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (p * o - 1.0) / (o - 1.0)
    return np.where(np.isfinite(f), np.maximum(f, 0.0), 0.0)
