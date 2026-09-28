"""Raw UFC Stats CSV -> clean Parquet tables.

Output tables (see docs/data_dictionary.md):
  events, fighters, fights, fight_stats, name_resolution, data_quality.json

Conventions
- Missing values stay NaN (never replaced by 0). "---"/"--" in the raw data mean missing.
- Statistics of a fight are summed over rounds; a fight without any stats row has
  has_stats = False and NaN statistics (distinct from "0 strikes landed").
- fighter_1 / fighter_2 keep the UFC Stats order, which is NOT random (winner is usually
  listed first). Downstream code must never use this order as information.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_quant.data.reconcile import resolve_fighter_ids

log = logging.getLogger(__name__)

STAT_PAIRS = {  # raw column -> output prefix, "x of y" format
    "SIG.STR.": "sig", "TOTAL STR.": "tot", "TD": "td", "HEAD": "head", "BODY": "body",
    "LEG": "leg", "DISTANCE": "distance", "CLINCH": "clinch", "GROUND": "ground",
}
STAT_COUNTS = {"KD": "kd", "SUB.ATT": "sub_att", "REV.": "rev"}


def _strip(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = df[c].astype("string").str.strip()
    return df


def _id_from_url(url: pd.Series) -> pd.Series:
    return url.astype("string").str.rstrip("/").str.split("/").str[-1]


def parse_height_cm(s: str) -> float:
    m = re.match(r"(\d+)'\s*(\d+)", str(s))
    return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1) if m else np.nan


def parse_inches_cm(s: str) -> float:
    m = re.match(r"(\d+(?:\.\d+)?)", str(s))
    return round(float(m.group(1)) * 2.54, 1) if m and str(s) != "--" else np.nan


def parse_mmss(s) -> float:
    m = re.match(r"^(\d+):(\d{2})$", str(s).strip()) if pd.notna(s) else None
    return float(int(m.group(1)) * 60 + int(m.group(2))) if m else np.nan


def parse_x_of_y(s) -> tuple[float, float]:
    m = re.match(r"^(\d+)\s+of\s+(\d+)$", str(s).strip()) if pd.notna(s) else None
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def parse_time_format(fmt: str) -> tuple[float, list[float] | None]:
    """'3 Rnd (5-5-5)' -> (3, [300, 300, 300]); unknown formats -> (nan, None)."""
    fmt = str(fmt)
    m = re.match(r"^(\d+)\s+Rnd", fmt)
    scheduled = float(m.group(1)) if m else np.nan
    d = re.search(r"\(([\d\-]+)\)", fmt)
    durations = [float(x) * 60 for x in d.group(1).split("-")] if d else None
    return scheduled, durations


def fight_seconds(end_round: float, end_time: float, durations: list[float] | None) -> float:
    if durations is None or np.isnan(end_round) or np.isnan(end_time):
        return np.nan
    r = int(end_round)
    if r < 1 or r > len(durations):
        return np.nan
    return float(sum(durations[: r - 1]) + end_time)


def transform_ufcstats(raw_dir: Path, processed_dir: Path) -> dict:
    src = Path(raw_dir) / "ufcstats"
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    quality: dict = {}

    ev = _strip(pd.read_csv(src / "ufc_event_details.csv", dtype=str))
    det = _strip(pd.read_csv(src / "ufc_fight_details.csv", dtype=str))
    res = _strip(pd.read_csv(src / "ufc_fight_results.csv", dtype=str))
    st = _strip(pd.read_csv(src / "ufc_fight_stats.csv", dtype=str))
    tott = _strip(pd.read_csv(src / "ufc_fighter_tott.csv", dtype=str))

    # ---- events
    events = pd.DataFrame({
        "event_id": _id_from_url(ev["URL"]),
        "event_name": ev["EVENT"],
        "event_date": pd.to_datetime(ev["DATE"], format="%B %d, %Y", errors="coerce"),
        "location": ev["LOCATION"],
    }).drop_duplicates("event_id")
    quality["events_duplicate_names"] = int(events["event_name"].duplicated().sum())

    # ---- fighters (tale of the tape = current snapshot; only static attributes are used)
    fighters = pd.DataFrame({
        "fighter_id": _id_from_url(tott["URL"]),
        "name": tott["FIGHTER"],
        "height_cm": tott["HEIGHT"].map(parse_height_cm),
        "reach_cm": tott["REACH"].map(parse_inches_cm),
        "weight_lbs": tott["WEIGHT"].str.extract(r"(\d+)")[0].astype(float),
        "stance": tott["STANCE"].replace({"": pd.NA}),
        "dob": pd.to_datetime(tott["DOB"], format="%b %d, %Y", errors="coerce"),
    }).drop_duplicates("fighter_id")

    # ---- fights
    res = res.drop_duplicates("URL")
    quality["results_duplicate_urls_dropped"] = int(len(_strip(pd.read_csv(src / "ufc_fight_results.csv", dtype=str))) - len(res))
    names = res["BOUT"].str.split(r"\s+vs\.\s+", n=1, expand=True, regex=True)
    fights = pd.DataFrame({
        "fight_id": _id_from_url(res["URL"]),
        "event_name": res["EVENT"],
        "bout": res["BOUT"],
        "fighter_1_name": names[0].str.strip(),
        "fighter_2_name": names[1].str.strip(),
        "outcome_raw": res["OUTCOME"],
        "weight_class": res["WEIGHTCLASS"],
        "method": res["METHOD"],
        "end_round": pd.to_numeric(res["ROUND"], errors="coerce"),
        "end_time_seconds": res["TIME"].map(parse_mmss),
        "time_format": res["TIME FORMAT"],
        "referee": res["REFEREE"],
    })
    fights["result"] = fights["outcome_raw"].map(
        {"W/L": "fighter_1", "L/W": "fighter_2", "D/D": "draw", "NC/NC": "no_contest"})
    tf = fights["time_format"].map(parse_time_format)
    fights["scheduled_rounds"] = [t[0] for t in tf]
    fights["fight_seconds"] = [fight_seconds(r, t, d[1]) for r, t, d in
                               zip(fights["end_round"], fights["end_time_seconds"], tf)]
    wc = fights["weight_class"].fillna("")
    fights["is_title"] = wc.str.contains("Title", case=False)
    fights["is_women"] = wc.str.contains("Women", case=False)
    ev_by_name = events.drop_duplicates("event_name", keep=False).set_index("event_name")
    fights = fights.join(ev_by_name[["event_id", "event_date"]], on="event_name")
    quality["fights_without_event_match"] = int(fights["event_id"].isna().sum())
    fights = fights.dropna(subset=["event_id", "event_date"])

    # fighter ids
    parts = pd.concat([
        fights[["fight_id", "event_date", "weight_class"]].assign(slot=1, name=fights["fighter_1_name"]),
        fights[["fight_id", "event_date", "weight_class"]].assign(slot=2, name=fights["fighter_2_name"]),
    ], ignore_index=True)
    resolved = resolve_fighter_ids(parts, fighters)
    name_resolution = (resolved.groupby(["name", "fighter_id", "id_status", "n_candidates"], dropna=False)
                       .size().rename("n_appearances").reset_index())
    for slot in (1, 2):
        r = resolved[resolved["slot"] == slot].set_index("fight_id")
        fights[f"fighter_{slot}_id"] = fights["fight_id"].map(r["fighter_id"])
        fights[f"fighter_{slot}_id_status"] = fights["fight_id"].map(r["id_status"])
    fights["id_ambiguous"] = (fights["fighter_1_id_status"] == "ambiguous") | (fights["fighter_2_id_status"] == "ambiguous")
    quality["id_status_counts"] = resolved["id_status"].value_counts().to_dict()
    quality["fights_with_ambiguous_fighter"] = int(fights["id_ambiguous"].sum())

    # add unresolved fighters (no physical attributes) so every id exists in `fighters`
    extra = resolved.loc[resolved["id_status"].isin(["unresolved", "ambiguous"]), ["fighter_id", "name"]].drop_duplicates("fighter_id")
    fighters["id_status"] = "ufcstats"
    extra = extra.assign(id_status=np.where(extra["fighter_id"].str.startswith("ambig_"), "ambiguous", "unresolved"))
    fighters = pd.concat([fighters, extra], ignore_index=True)

    # ---- fight stats (sum over rounds, attach to fighter slot within the fight)
    det = det.drop_duplicates()
    dup_keys = det[det.duplicated(["EVENT", "BOUT"], keep=False)]
    quality["stats_ambiguous_event_bout_keys"] = int(dup_keys[["EVENT", "BOUT"]].drop_duplicates().shape[0])
    det_u = det.drop_duplicates(["EVENT", "BOUT"], keep=False)
    det_u = det_u.assign(fight_id=_id_from_url(det_u["URL"]))
    st = st.dropna(subset=["FIGHTER"]).merge(det_u[["EVENT", "BOUT", "fight_id"]], on=["EVENT", "BOUT"], how="inner")
    parsed = {"fight_id": st["fight_id"], "fighter_name": st["FIGHTER"]}
    for col, pre in STAT_PAIRS.items():
        xy = st[col].map(parse_x_of_y)
        parsed[f"{pre}_landed"] = [a for a, _ in xy]
        parsed[f"{pre}_att"] = [b for _, b in xy]
    for col, name in STAT_COUNTS.items():
        parsed[name] = pd.to_numeric(st[col], errors="coerce")
    parsed["ctrl_seconds"] = st["CTRL"].map(parse_mmss)
    rounds = pd.DataFrame(parsed)
    stat_cols = [c for c in rounds.columns if c not in ("fight_id", "fighter_name")]
    # min_count=1 keeps NaN when every round is missing instead of summing to 0
    fs = rounds.groupby(["fight_id", "fighter_name"])[stat_cols].sum(min_count=1).reset_index()
    slot_map = pd.concat([
        fights[["fight_id", "fighter_1_name", "fighter_1_id"]].set_axis(["fight_id", "fighter_name", "fighter_id"], axis=1),
        fights[["fight_id", "fighter_2_name", "fighter_2_id"]].set_axis(["fight_id", "fighter_name", "fighter_id"], axis=1),
    ])
    fs = fs.merge(slot_map, on=["fight_id", "fighter_name"], how="inner")
    fs = fs.drop_duplicates(["fight_id", "fighter_id"])
    has_both = fs.groupby("fight_id")["fighter_id"].transform("count") == 2
    quality["stats_rows_dropped_one_sided"] = int((~has_both).sum())
    fs = fs[has_both]
    fights["has_stats"] = fights["fight_id"].isin(fs["fight_id"])
    quality["fights_with_stats"] = int(fights["has_stats"].sum())

    fights = fights.sort_values(["event_date", "event_id", "fight_id"]).reset_index(drop=True)
    quality["n_events"] = int(fights["event_id"].nunique())
    quality["n_fights"] = int(len(fights))
    quality["result_counts"] = fights["result"].value_counts(dropna=False).to_dict()
    quality["date_min"] = str(fights["event_date"].min().date())
    quality["date_max"] = str(fights["event_date"].max().date())
    quality["fighter_missing_rates"] = fighters.loc[fighters["id_status"] == "ufcstats",
                                                    ["height_cm", "reach_cm", "dob", "stance"]].isna().mean().round(4).to_dict()

    events.to_parquet(out / "events.parquet", index=False)
    fighters.to_parquet(out / "fighters.parquet", index=False)
    fights.to_parquet(out / "fights.parquet", index=False)
    fs.to_parquet(out / "fight_stats.parquet", index=False)
    name_resolution.to_parquet(out / "name_resolution.parquet", index=False)
    (out / "data_quality_ufcstats.json").write_text(json.dumps(quality, indent=2, default=str))
    log.info("transform done: %s", quality)
    return quality
