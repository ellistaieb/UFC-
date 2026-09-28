"""Identifier reconciliation: fighter names -> UFC Stats ids, odds rows -> fights.

Ambiguous or failed matches are never validated silently: every decision is written
to a reconciliation table with an explicit status.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

WEIGHT_LIMITS_LBS = {
    "strawweight": 115, "flyweight": 125, "bantamweight": 135, "featherweight": 145,
    "lightweight": 155, "welterweight": 170, "middleweight": 185,
    "light heavyweight": 205, "heavyweight": 265,
}


# letters that Unicode NFKD does not decompose into an ASCII base letter
_TRANSLIT = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss",
                           "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i"})


def normalize_name(name: str) -> str:
    """Lowercase, transliterate, strip accents/punctuation/suffixes, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name.translate(_TRANSLIT)).encode("ascii", "ignore").decode()
    s = s.lower().replace("-", " ").replace(".", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def weight_class_limit(weight_class: str) -> float:
    wc = (weight_class or "").lower()
    for key in sorted(WEIGHT_LIMITS_LBS, key=len, reverse=True):
        if key in wc:
            return float(WEIGHT_LIMITS_LBS[key])
    return np.nan


def resolve_fighter_ids(participants: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Map each (fight, name) appearance to a fighter_id.

    participants: fight_id, event_date, weight_class, name
    fighters: fighter_id, name, weight_lbs, dob
    Returns participants with fighter_id and id_status in
    {"unique", "disambiguated", "ambiguous", "unresolved"}.
    """
    fighters = fighters.assign(name_norm=fighters["name"].map(normalize_name))
    by_name = fighters.groupby("name_norm")
    out = participants.copy()
    out["name_norm"] = out["name"].map(normalize_name)
    ids, statuses, n_cands = [], [], []
    for row in out.itertuples(index=False):
        cands = by_name.get_group(row.name_norm) if row.name_norm in by_name.groups else None
        if cands is None:
            ids.append("unres_" + row.name_norm.replace(" ", "_"))
            statuses.append("unresolved")
            n_cands.append(0)
            continue
        n_cands.append(len(cands))
        if len(cands) == 1:
            ids.append(cands["fighter_id"].iloc[0])
            statuses.append("unique")
            continue
        keep = cands
        limit = weight_class_limit(row.weight_class)
        if not np.isnan(limit):
            # fighter's listed weight should be within one class-ish of the bout limit
            close = keep[(keep["weight_lbs"] - limit).abs() <= 15]
            if len(close) >= 1:
                keep = close
        if len(keep) > 1 and pd.notna(row.event_date):
            age = (row.event_date - keep["dob"]).dt.days / 365.25
            plausible = keep[age.between(18, 48) | keep["dob"].isna()]
            if len(plausible) >= 1:
                keep = plausible
        if len(keep) == 1:
            ids.append(keep["fighter_id"].iloc[0])
            statuses.append("disambiguated")
        else:
            ids.append("ambig_" + row.name_norm.replace(" ", "_"))
            statuses.append("ambiguous")
    out["fighter_id"] = ids
    out["id_status"] = statuses
    out["n_candidates"] = n_cands
    return out


def match_odds_to_fights(odds_pairs: pd.DataFrame, fights: pd.DataFrame, fighters: pd.DataFrame,
                         tolerance_days: int = 1) -> pd.DataFrame:
    """Match two-sided odds rows (by names and date) to UFC Stats fights.

    odds_pairs: source_row, event_date, name_a, name_b, ...
    fights: fight_id, event_date, fighter_1_id, fighter_2_id, fighter_1_name, fighter_2_name
    Returns one row per odds pair with fight_id, fighter ids for name_a/name_b and match_status
    in {"matched", "ambiguous", "unmatched"}.
    """
    f = fights[["fight_id", "event_date", "fighter_1_id", "fighter_2_id",
                "fighter_1_name", "fighter_2_name"]].copy()
    f["n1"] = f["fighter_1_name"].map(normalize_name)
    f["n2"] = f["fighter_2_name"].map(normalize_name)
    # index fights by unordered normalized name pair
    f["pair"] = [frozenset((a, b)) for a, b in zip(f["n1"], f["n2"])]
    pair_index: dict[frozenset, list[int]] = {}
    for i, p in enumerate(f["pair"]):
        pair_index.setdefault(p, []).append(i)
    # fallback index by last-name pair (handles "Alex" vs "Alexander" style differences)
    f["last_pair"] = [frozenset((a.split(" ")[-1] if a else "", b.split(" ")[-1] if b else ""))
                      for a, b in zip(f["n1"], f["n2"])]
    last_index: dict[frozenset, list[int]] = {}
    for i, p in enumerate(f["last_pair"]):
        last_index.setdefault(p, []).append(i)

    tol = pd.Timedelta(days=tolerance_days)
    rows = []
    for r in odds_pairs.itertuples(index=False):
        na, nb = normalize_name(r.name_a), normalize_name(r.name_b)
        method = "full_name"
        cand = [i for i in pair_index.get(frozenset((na, nb)), [])
                if abs(f["event_date"].iat[i] - r.event_date) <= tol]
        if not cand:
            method = "last_name"
            lp = frozenset((na.split(" ")[-1], nb.split(" ")[-1]))
            cand = [i for i in last_index.get(lp, []) if abs(f["event_date"].iat[i] - r.event_date) <= tol]
        rec = {"source_row": r.source_row, "match_method": method, "n_candidates": len(cand),
               "fight_id": None, "fighter_a_id": None, "fighter_b_id": None}
        if len(cand) == 1:
            i = cand[0]
            n1 = f["n1"].iat[i]
            a_is_1 = (na == n1) if method == "full_name" else (na.split(" ")[-1] == n1.split(" ")[-1])
            if method == "last_name" and na.split(" ")[-1] == nb.split(" ")[-1]:
                rec["match_status"] = "ambiguous"  # same last names: orientation unknowable
            else:
                rec.update(match_status="matched", fight_id=f["fight_id"].iat[i],
                           fighter_a_id=f["fighter_1_id"].iat[i] if a_is_1 else f["fighter_2_id"].iat[i],
                           fighter_b_id=f["fighter_2_id"].iat[i] if a_is_1 else f["fighter_1_id"].iat[i])
        elif len(cand) > 1:
            rec["match_status"] = "ambiguous"
        else:
            rec["match_status"] = "unmatched"
        rows.append(rec)
    res = pd.DataFrame(rows)
    return odds_pairs.merge(res, on="source_row", how="left")
