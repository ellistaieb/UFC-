"""SYNTHETIC demonstration data (NOT real UFC data).

Generates raw tables in the same schema as the UFC Stats mirror, plus odds in the standard
long format with exact timestamps, so the entire pipeline (including the decision-time
filter) can be exercised offline. Every name contains "SYNTHETIC"/"Synth" and outputs are
written under data/synthetic/ only.

Generative model: latent skill per fighter (random walk), P(win) = logistic(skill diff),
striking/grappling stats noisy functions of skill, bookmaker price = noisy estimate of the
true probability plus a margin, quoted at several times (some after the decision time).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ufc_quant.config import Paths
from ufc_quant.data.odds import import_odds_csv

WEIGHT_CLASSES = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
                  "Middleweight", "Light Heavyweight", "Heavyweight"]
WC_LBS = [125, 135, 145, 155, 170, 185, 205, 265]


def _hex(rng, n=16):
    return "".join(rng.choice(list("0123456789abcdef"), size=n))


def write_synthetic_raw(cfg: dict, paths: Paths) -> dict:
    s = cfg["synthetic"]
    rng = np.random.default_rng(s["seed"])
    raw = paths.raw / "ufcstats"
    raw.mkdir(parents=True, exist_ok=True)
    n_f = s["n_fighters"]
    fighters = pd.DataFrame({
        "fid": [_hex(rng) for _ in range(n_f)],
        "name": [f"Synth Fighter {i:04d}" for i in range(n_f)],
        "wc": rng.integers(0, len(WEIGHT_CLASSES), n_f),
        "skill": rng.normal(0, 1, n_f),
        "dob": pd.to_datetime("1975-01-01") + pd.to_timedelta(rng.integers(0, 365 * 25, n_f), unit="D"),
        "height_in": rng.normal(70, 3, n_f).round(),
        "reach_extra": rng.normal(1.5, 2, n_f).round(),
    })
    dates = pd.date_range(s["start_date"], s["end_date"], freq=f"{s['days_between_events']}D")
    ev_rows, det_rows, res_rows, st_rows, odds_rows = [], [], [], [], []
    for e_i, d in enumerate(dates):
        ev_name = f"SYNTHETIC Event {e_i:04d}"
        ev_url = f"http://synthetic.local/event-details/{_hex(rng)}"
        ev_rows.append({"EVENT": ev_name, "URL": ev_url, "DATE": d.strftime("%B %d, %Y"), "LOCATION": "SYNTHETIC"})
        # mean-reverting skill drift between events (keeps the skill dispersion stationary)
        fighters["skill"] = 0.99 * fighters["skill"] + rng.normal(0, 0.14, n_f)
        used = set()
        for k in range(s["fights_per_event"]):
            wc = rng.integers(0, len(WEIGHT_CLASSES))
            pool = fighters.index[(fighters["wc"] == wc) & ~fighters.index.isin(list(used))]
            if len(pool) < 2:
                continue
            i, j = rng.choice(pool, 2, replace=False)
            used |= {i, j}
            fi, fj = fighters.loc[i], fighters.loc[j]
            age_i = (d - fi["dob"]).days / 365.25
            age_j = (d - fj["dob"]).days / 365.25
            logit_true = 1.0 * (fi["skill"] - fj["skill"]) - 0.04 * (age_i - age_j)
            p_i = 1 / (1 + np.exp(-logit_true))
            u = rng.random()
            nc = rng.random() < 0.01
            draw = (not nc) and rng.random() < 0.008
            i_wins = u < p_i
            outcome = "NC/NC" if nc else "D/D" if draw else ("W/L" if i_wins else "L/W")
            five = rng.random() < 0.08
            n_sched = 5 if five else 3
            end_round = int(rng.integers(1, n_sched + 1)) if rng.random() < 0.5 else n_sched
            end_time = 300 if end_round == n_sched else int(rng.integers(10, 300))
            bout = f"{fi['name']} vs. {fj['name']}"
            f_url = f"http://synthetic.local/fight-details/{_hex(rng)}"
            det_rows.append({"EVENT": ev_name, "BOUT": bout, "URL": f_url})
            res_rows.append({"EVENT": ev_name, "BOUT": bout, "OUTCOME": outcome,
                             "WEIGHTCLASS": f"{WEIGHT_CLASSES[wc]} Bout", "METHOD": "Decision - Unanimous",
                             "ROUND": end_round, "TIME": f"{end_time // 60}:{end_time % 60:02d}",
                             "TIME FORMAT": "5 Rnd (5-5-5-5-5)" if five else "3 Rnd (5-5-5)",
                             "REFEREE": "SYNTHETIC", "DETAILS": "", "URL": f_url})
            for rnd in range(1, end_round + 1):
                secs = 300 if rnd < end_round else end_time
                for me, opp in ((fi, fj), (fj, fi)):
                    edge = me["skill"] - opp["skill"]
                    att = max(0, int(rng.poisson(secs / 60 * 8)))
                    acc = np.clip(0.45 + 0.05 * edge + rng.normal(0, 0.05), 0.05, 0.95)
                    landed = int(rng.binomial(att, acc))
                    td_att = int(rng.poisson(secs / 300 * 1.2))
                    td_l = int(rng.binomial(td_att, np.clip(0.4 + 0.05 * edge, 0.05, 0.95)))
                    ctrl = int(min(secs, rng.poisson(20 * (1 + max(edge, 0)))))
                    st_rows.append({"EVENT": ev_name, "BOUT": bout, "ROUND": f"Round {rnd}", "FIGHTER": me["name"],
                                    "KD": int(rng.random() < 0.03), "SIG.STR.": f"{landed} of {att}", "SIG.STR. %": "",
                                    "TOTAL STR.": f"{landed} of {att}", "TD": f"{td_l} of {td_att}", "TD %": "",
                                    "SUB.ATT": int(rng.random() < 0.1), "REV.": 0, "CTRL": f"{ctrl // 60}:{ctrl % 60:02d}",
                                    "HEAD": f"{landed} of {att}", "BODY": "0 of 0", "LEG": "0 of 0",
                                    "DISTANCE": f"{landed} of {att}", "CLINCH": "0 of 0", "GROUND": "0 of 0"})
            # bookmaker: noisy view of the truth (sharper than a naive model) + ~4.5% margin
            commence = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=20, minutes=30 * k)
            for label, offset_h, noise in (("open", -120, 0.45), ("t-6h", -6, 0.30), ("close", -0.5, 0.25),
                                           ("in_play", 0.2, 0.05)):
                l_m = logit_true + rng.normal(0, noise)
                pm = float(np.clip(1 / (1 + np.exp(-l_m)), 0.03, 0.93))
                margin = 1.045
                for fid, pp in ((fi["fid"], pm), (fj["fid"], 1 - pm)):
                    odds_rows.append({"fight_url": f_url, "fighter_id": fid, "bookmaker": "SYNTHETIC_BOOK",
                                      "snapshot_timestamp": (commence + pd.Timedelta(hours=offset_h)).isoformat(),
                                      "commence_timestamp": commence.isoformat(),
                                      "decimal_odds": round(1 / (pp * margin), 3), "snapshot_label": label})
    tott = pd.DataFrame({
        "FIGHTER": fighters["name"],
        "HEIGHT": [f"{int(h // 12)}' {int(h % 12)}\"" for h in fighters["height_in"]],
        "WEIGHT": [f"{WC_LBS[w]} lbs." for w in fighters["wc"]],
        "REACH": [f"{int(h + r)}\"" for h, r in zip(fighters["height_in"], fighters["reach_extra"])],
        "STANCE": rng.choice(["Orthodox", "Southpaw"], n_f),
        "DOB": fighters["dob"].dt.strftime("%b %d, %Y"),
        "URL": "http://synthetic.local/fighter-details/" + fighters["fid"],
    })
    pd.DataFrame(ev_rows).to_csv(raw / "ufc_event_details.csv", index=False)
    pd.DataFrame(det_rows).to_csv(raw / "ufc_fight_details.csv", index=False)
    pd.DataFrame(res_rows).to_csv(raw / "ufc_fight_results.csv", index=False)
    pd.DataFrame(st_rows).to_csv(raw / "ufc_fight_stats.csv", index=False)
    tott.to_csv(raw / "ufc_fighter_tott.csv", index=False)
    tott[["FIGHTER", "URL"]].to_csv(raw / "ufc_fighter_details.csv", index=False)
    odds_dir = paths.raw / "odds"
    odds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(odds_rows).to_csv(odds_dir / "synthetic_odds.csv", index=False)
    manifest = {"source": "SYNTHETIC generator (ufc_quant.data.synthetic)", "seed": s["seed"],
                "n_events": len(ev_rows), "n_fights": len(res_rows), "WARNING": "SYNTHETIC DATA - NOT REAL"}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_synthetic_odds(paths: Paths) -> dict:
    raw = pd.read_csv(paths.raw / "odds" / "synthetic_odds.csv")
    raw["fight_id"] = raw["fight_url"].str.rstrip("/").str.split("/").str[-1]
    odds = import_odds_csv_frame(raw.assign(source="SYNTHETIC"))
    odds.to_parquet(paths.processed / "odds.parquet", index=False)
    rep = {"SYNTHETIC": {"rows": int(len(odds)), "snapshot_precision": odds["snapshot_precision"].value_counts().to_dict()}}
    (paths.processed / "data_quality_odds.json").write_text(json.dumps(rep, indent=2))
    return rep


def import_odds_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return import_odds_csv(buf)
