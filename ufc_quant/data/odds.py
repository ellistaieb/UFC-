"""Odds ingestion into a standard long schema, pre-decision selection and pairing.

Standard schema (one row per fight x fighter x bookmaker x snapshot):
    fight_id, fighter_id, bookmaker, snapshot_timestamp, commence_timestamp,
    decimal_odds, snapshot_precision, source

snapshot_precision is one of:
    "exact"     snapshot_timestamp is a real observation time
    "date"      only the calendar date of the observation is known
    "unknown"   the source does not document when the odds were observed
                (at best: some time up to the fight, possibly closing odds)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_quant.backtest.finance import american_to_decimal, devig_proportional, overround
from ufc_quant.data.http import PoliteDownloader
from ufc_quant.data.reconcile import match_odds_to_fights

log = logging.getLogger(__name__)

STANDARD_COLUMNS = ["fight_id", "fighter_id", "bookmaker", "snapshot_timestamp",
                    "commence_timestamp", "decimal_odds", "snapshot_precision", "source"]
REQUIRED_CSV_COLUMNS = ["fight_id", "fighter_id", "bookmaker", "snapshot_timestamp", "decimal_odds"]


# --------------------------------------------------------------------------- CSV import
def validate_standard_odds(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"odds CSV missing required columns: {missing}")
    out = df.copy()
    if "commence_timestamp" not in out:
        out["commence_timestamp"] = pd.NaT
    out["snapshot_timestamp"] = pd.to_datetime(out["snapshot_timestamp"], utc=True, errors="coerce")
    out["commence_timestamp"] = pd.to_datetime(out["commence_timestamp"], utc=True, errors="coerce")
    out["decimal_odds"] = pd.to_numeric(out["decimal_odds"], errors="coerce")
    bad = ~(out["decimal_odds"] > 1.0)
    if bad.any():
        log.warning("dropping %d rows with invalid decimal odds (<= 1 or missing)", int(bad.sum()))
        out = out[~bad]
    if "snapshot_precision" not in out:
        out["snapshot_precision"] = np.where(out["snapshot_timestamp"].notna(), "exact", "unknown")
    if "source" not in out:
        out["source"] = "csv_import"
    return out[STANDARD_COLUMNS]


def import_odds_csv(path: str | Path) -> pd.DataFrame:
    return validate_standard_odds(pd.read_csv(path, dtype={"fight_id": str, "fighter_id": str}))


# --------------------------------------------------------------------------- adapters
def _pairs_ultimate(df: pd.DataFrame) -> pd.DataFrame:
    """ultimate_ufc_dataset: American odds for red/blue corners, date only."""
    return pd.DataFrame({
        "event_date": pd.to_datetime(df["date"], errors="coerce"),
        "name_a": df["R_fighter"], "name_b": df["B_fighter"],
        "odds_a": american_to_decimal(pd.to_numeric(df["R_odds"], errors="coerce")),
        "odds_b": american_to_decimal(pd.to_numeric(df["B_odds"], errors="coerce")),
    })


def _pairs_betmma(df: pd.DataFrame) -> pd.DataFrame:
    """jansen88/ufc-data (betmma.tips odds): favourite/underdog decimal odds, date only."""
    return pd.DataFrame({
        "event_date": pd.to_datetime(df["event_date"], errors="coerce"),
        "name_a": df["favourite"], "name_b": df["underdog"],
        "odds_a": pd.to_numeric(df["favourite_odds"], errors="coerce"),
        "odds_b": pd.to_numeric(df["underdog_odds"], errors="coerce"),
    })


ADAPTERS = {"ultimate_ufc_dataset": ("ultimate_ufc_dataset.csv", _pairs_ultimate),
            "betmma_tips": ("betmma_tips_jansen88.csv", _pairs_betmma)}


def acquire_public_odds(cfg: dict, raw_dir: Path, refresh: bool = False) -> dict:
    dl = PoliteDownloader.from_config(cfg)
    out = Path(raw_dir) / "odds"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, (fname, _) in ADAPTERS.items():
        scfg = cfg["sources"]["odds"].get(key, {})
        if not scfg.get("enabled"):
            continue
        try:
            manifest[key] = dl.fetch_to_file(scfg["url"], out / fname, refresh=refresh)
        except RuntimeError as exc:
            log.warning("odds source %s unavailable: %s", key, exc)
            manifest[key] = {"error": str(exc)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_odds_table(cfg: dict, raw_dir: Path, processed_dir: Path) -> dict:
    """Convert every available odds source to the standard schema, with a reconciliation table."""
    fights = pd.read_parquet(Path(processed_dir) / "fights.parquet")
    fighters = pd.read_parquet(Path(processed_dir) / "fighters.parquet")
    tol = cfg["reconciliation"]["date_tolerance_days"]
    long_frames, recon_frames, report = [], [], {}

    for key, (fname, adapter) in ADAPTERS.items():
        scfg = cfg["sources"]["odds"].get(key, {})
        path = Path(raw_dir) / "odds" / fname
        if not scfg.get("enabled") or not path.exists():
            report[key] = {"status": "not available"}
            continue
        raw = pd.read_csv(path, low_memory=False)
        pairs = adapter(raw)
        pairs["source_row"] = np.arange(len(pairs))
        pairs = pairs[np.isfinite(pairs["odds_a"]) & np.isfinite(pairs["odds_b"])
                      & (pairs["odds_a"] > 1) & (pairs["odds_b"] > 1) & pairs["event_date"].notna()]
        matched = match_odds_to_fights(pairs, fights, fighters, tolerance_days=tol)
        matched["source"] = key
        recon_frames.append(matched)
        ok = matched[matched["match_status"] == "matched"].drop_duplicates("fight_id", keep=False)
        dup_fights = matched[matched["match_status"] == "matched"]["fight_id"].duplicated(keep=False).sum()
        bookmaker = scfg["bookmaker_label"]
        for side in ("a", "b"):
            long_frames.append(pd.DataFrame({
                "fight_id": ok["fight_id"].values, "fighter_id": ok[f"fighter_{side}_id"].values,
                "bookmaker": bookmaker, "snapshot_timestamp": pd.NaT, "commence_timestamp": pd.NaT,
                "decimal_odds": ok[f"odds_{side}"].values, "snapshot_precision": "unknown", "source": key,
            }))
        report[key] = {
            "rows_with_two_sided_odds": int(len(pairs)),
            "match_status": matched["match_status"].value_counts().to_dict(),
            "matched_fights_used": int(len(ok)),
            "fights_matched_by_several_rows_dropped": int(dup_fights),
            "date_min": str(fights.loc[fights["fight_id"].isin(ok["fight_id"]), "event_date"].min().date()) if len(ok) else None,
            "date_max": str(fights.loc[fights["fight_id"].isin(ok["fight_id"]), "event_date"].max().date()) if len(ok) else None,
        }

    for p in cfg["sources"]["odds"].get("csv_imports", []) or []:
        long_frames.append(import_odds_csv(p))
        report[f"csv:{p}"] = {"status": "imported"}

    odds = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame(columns=STANDARD_COLUMNS)
    odds["snapshot_timestamp"] = pd.to_datetime(odds["snapshot_timestamp"], utc=True)
    odds["commence_timestamp"] = pd.to_datetime(odds["commence_timestamp"], utc=True)
    odds.to_parquet(Path(processed_dir) / "odds.parquet", index=False)
    if recon_frames:
        recon = pd.concat(recon_frames, ignore_index=True)
        recon.to_parquet(Path(processed_dir) / "odds_reconciliation.parquet", index=False)
    (Path(processed_dir) / "data_quality_odds.json").write_text(json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------------------- selection
def select_pre_decision_odds(odds: pd.DataFrame, fights: pd.DataFrame, decision_hours_before: float,
                             accept_imprecise: bool) -> pd.DataFrame:
    """Keep, per fight x fighter x bookmaker, the latest snapshot available at decision time.

    - exact timestamps: snapshot <= commence - decision_hours_before. If the commence time is
      unknown, the conservative rule uses the event date at 00:00 UTC as commence time.
    - date-only / unknown precision: accepted only if accept_imprecise, never "filled" with
      a later observation. These rows are flagged via snapshot_precision.
    A missing quote is never replaced by a future one.
    """
    if odds.empty:
        return odds.copy()
    o = odds.merge(fights[["fight_id", "event_date"]], on="fight_id", how="inner")
    event_midnight = pd.to_datetime(o["event_date"]).dt.tz_localize("UTC")
    commence = o["commence_timestamp"].fillna(event_midnight)
    decision = commence - pd.Timedelta(hours=decision_hours_before)
    exact = o["snapshot_precision"] == "exact"
    ok_exact = exact & o["snapshot_timestamp"].notna() & (o["snapshot_timestamp"] <= decision)
    if "date" in set(o["snapshot_precision"]):
        # date-only snapshot: must be strictly before the event date to be provably pre-decision
        snap_day = o["snapshot_timestamp"].dt.normalize()
        ok_date = (o["snapshot_precision"] == "date") & (snap_day < event_midnight)
    else:
        ok_date = pd.Series(False, index=o.index)
    ok_imprecise = (o["snapshot_precision"] == "unknown") & accept_imprecise
    if accept_imprecise:
        ok_date = ok_date | (o["snapshot_precision"] == "date")
    kept = o[ok_exact | ok_date | ok_imprecise].copy()
    kept["_order"] = kept["snapshot_timestamp"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    kept = kept.sort_values("_order").groupby(["fight_id", "fighter_id", "bookmaker"], as_index=False).tail(1)
    return kept.drop(columns=["_order", "event_date"])


def pair_two_sided(odds: pd.DataFrame, fights_oriented: pd.DataFrame) -> pd.DataFrame:
    """Build one row per fight and bookmaker with both sides from the SAME snapshot.

    fights_oriented: fight_id, fighter_a_id, fighter_b_id (canonical orientation).
    Returns fight_id, bookmaker, odds_a, odds_b, overround, p_market_a (proportional de-vig),
    snapshot_precision.
    """
    cols = ["fight_id", "bookmaker", "odds_a", "odds_b", "overround", "p_market_a", "snapshot_precision"]
    if odds.empty:
        return pd.DataFrame(columns=cols)
    f = fights_oriented[["fight_id", "fighter_a_id", "fighter_b_id"]]
    o = odds.merge(f, on="fight_id")
    snap = o["snapshot_timestamp"].astype("string").fillna("NA")
    o = o.assign(_snap=snap)
    a = o[o["fighter_id"] == o["fighter_a_id"]]
    b = o[o["fighter_id"] == o["fighter_b_id"]]
    m = a.merge(b, on=["fight_id", "bookmaker", "_snap"], suffixes=("_a", "_b"))
    out = pd.DataFrame({
        "fight_id": m["fight_id"], "bookmaker": m["bookmaker"],
        "odds_a": m["decimal_odds_a"], "odds_b": m["decimal_odds_b"],
        "snapshot_precision": m["snapshot_precision_a"],
    })
    out["overround"] = overround(out["odds_a"], out["odds_b"])
    out["p_market_a"] = devig_proportional(out["odds_a"], out["odds_b"])[0]
    return out[cols].drop_duplicates(["fight_id", "bookmaker"], keep="last")


# --------------------------------------------------------------------------- The Odds API
class TheOddsAPIClient:
    """Minimal client for The Odds API v4 (historical endpoint).

    Not exercised during development: the host was blocked by the network proxy and
    historical odds require a paid plan. Endpoint and parameters follow the public v4
    documentation (to be re-checked before use):
        GET {base}/historical/sports/{sport}/odds?apiKey=..&regions=..&markets=h2h&date=ISO8601
    The API key is read from the ODDS_API_KEY environment variable (see .env.example).
    """

    def __init__(self, cfg: dict):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:  # pragma: no cover
            pass
        self.key = os.environ.get("ODDS_API_KEY", "").strip()
        self.cfg = cfg["sources"]["odds"]["the_odds_api"]
        self.dl = PoliteDownloader.from_config(cfg)

    def available(self) -> bool:
        return bool(self.key)

    def historical_snapshot(self, iso_timestamp: str) -> list[dict]:
        if not self.available():
            raise RuntimeError("ODDS_API_KEY absent : renseigner .env (voir .env.example)")
        url = f"{self.cfg['base_url']}/historical/sports/{self.cfg['sport_key']}/odds"
        params = {"apiKey": self.key, "regions": self.cfg["regions"], "markets": self.cfg["markets"],
                  "oddsFormat": "decimal", "date": iso_timestamp}
        self.dl._wait_for_host(url)
        r = self.dl.session.get(url, params=params, timeout=self.dl.timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    @staticmethod
    def to_long(snapshot_payload: list[dict], snapshot_ts: str) -> pd.DataFrame:
        """Flatten API events into (names, bookmaker, timestamps, odds). fight_id/fighter_id
        must then be resolved with the reconciliation module (names -> UFC Stats ids)."""
        rows = []
        for ev in snapshot_payload:
            for bm in ev.get("bookmakers", []):
                for mk in bm.get("markets", []):
                    if mk.get("key") != "h2h":
                        continue
                    for oc in mk.get("outcomes", []):
                        rows.append({"home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
                                     "fighter_name": oc.get("name"), "bookmaker": bm.get("key"),
                                     "snapshot_timestamp": mk.get("last_update") or snapshot_ts,
                                     "commence_timestamp": ev.get("commence_time"),
                                     "decimal_odds": oc.get("price"), "snapshot_precision": "exact",
                                     "source": "the_odds_api"})
        return pd.DataFrame(rows)
