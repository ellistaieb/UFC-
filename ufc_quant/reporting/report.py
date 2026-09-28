"""Markdown report generated exclusively from computed artifacts (no hand-written numbers)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ufc_quant.config import Paths

MODEL_LABELS = {"baseline": "Référence 50 %", "elo": "Elo", "logistic": "Régression logistique",
                "logistic_cal": "Logistique calibrée", "gbm": "Gradient boosting", "gbm_cal": "GBM calibré",
                "market": "Marché (cotes dé-margées)", "blend": "Mélange marché + modèle"}
STRAT_LABELS = {"flat": "Mise fixe (10 u.)", "fixed_fraction": "Fraction fixe (0,5 %)",
                "quarter_kelly": "¼ Kelly plafonné"}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _f(x, nd=4):
    if x is None:
        return "—"
    try:
        if x != x:  # NaN
            return "—"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _pct(x, nd=1):
    return "—" if x is None or x != x else f"{100 * x:.{nd}f} %"


def _table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_report(cfg: dict, paths: Paths) -> Path:
    art, proc = paths.artifacts, paths.processed
    meta = _load(art / "model_meta.json") or {}
    ev = _load(art / "evaluation.json") or {}
    bt = _load(art / "backtest" / "backtest_summary.json") or {}
    dmeta = _load(proc / "dataset_meta.json") or {}
    q_ufc = _load(proc / "data_quality_ufcstats.json") or {}
    q_odds = _load(proc / "data_quality_odds.json") or {}
    synthetic = paths.dataset_kind == "synthetic"
    L = []
    if synthetic:
        L.append("> ⚠️ **DÉMONSTRATION SYNTHÉTIQUE** — toutes les données de ce rapport sont générées "
                 "artificiellement. Aucun chiffre ne décrit l'UFC ni un marché réel.\n")
    L.append(f"# UFC Quant — rapport {'(SYNTHÉTIQUE) ' if synthetic else ''}généré automatiquement\n")
    L.append(f"- Généré le : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    L.append(f"- Version du modèle : {meta.get('model_version')} — entraîné le {meta.get('trained_at')}")
    L.append(f"- Données : {dmeta.get('data_min_date')} → {dmeta.get('data_max_date')} "
             f"({'synthétiques' if synthetic else 'réelles'})")
    L.append("- Tous les chiffres ci-dessous proviennent des fichiers `artifacts/` produits par le pipeline.\n")

    # --- data
    L.append("## 1. Données\n")
    L.append(f"- Combats : {q_ufc.get('n_fights')} sur {q_ufc.get('n_events')} événements "
             f"({q_ufc.get('date_min')} → {q_ufc.get('date_max')}).")
    L.append(f"- Issues : {q_ufc.get('result_counts')}.")
    L.append(f"- Combats avec statistiques détaillées : {q_ufc.get('fights_with_stats')}.")
    L.append(f"- Résolution des identifiants combattants : {q_ufc.get('id_status_counts')} ; "
             f"combats exclus pour identité ambiguë : {q_ufc.get('fights_with_ambiguous_fighter')}.")
    L.append(f"- Taux de valeurs manquantes (profils) : {q_ufc.get('fighter_missing_rates')}.")
    for k, v in q_odds.items():
        L.append(f"- Cotes `{k}` : {v}")
    cov = ev.get("coverage", {})
    L.append(f"- Échantillons de modélisation : validation {cov.get('n_validation')} combats décidés, "
             f"test {cov.get('n_test')} ; nuls {cov.get('n_draw')}, no contests {cov.get('n_no_contest')} "
             "(exclus du score, inclus dans le backtest).\n")

    # --- protocol
    L.append("## 2. Protocole\n")
    L.append(f"- Entraînement à partir du {meta.get('train_start')} ; validation walk-forward annuelle "
             f"{meta.get('walk_forward_years')} ; **test final intact à partir du {meta.get('test_start')}**.")
    L.append(f"- Elo : K = {meta.get('elo_k')} choisi sur les années de validation.")
    sel = meta.get("selection", {})
    L.append("- Hyperparamètres retenus (log-loss de validation) : " +
             "; ".join(f"{MODEL_LABELS.get(k, k)} {v.get('params')}" for k, v in sel.items()))
    L.append(f"- Températures de calibration (ajustées sur la validation) : {meta.get('temperatures')}")
    L.append(f"- Modèle utilisé pour le backtest (meilleure log-loss de validation) : "
             f"**{MODEL_LABELS.get(meta.get('betting_model'), meta.get('betting_model'))}**.\n")

    # --- quality
    L.append("## 3. Qualité des probabilités (tous les combats décidés)\n")
    rows = []
    for m in ev.get("metrics", []):
        rows.append([m["split"], MODEL_LABELS.get(m["model"], m["model"]), m["n"], _f(m["log_loss"]), _f(m["brier"]),
                     _f(m["ece"], 3), _f(m["accuracy"], 3), _f(m["roc_auc"], 3)])
    rows.sort(key=lambda r: (r[0] != "validation", r[1]))
    L.append(_table(rows, ["Segment", "Modèle", "n", "Log-loss", "Brier", "ECE", "Accuracy", "ROC-AUC"]))
    L.append("\nRéférence : log-loss d'une prédiction à 50 % = 0,6931 ; Brier = 0,25.\n")

    # --- market
    L.append("## 4. Valeur ajoutée par rapport au marché (mêmes combats)\n")
    if not ev.get("market_available"):
        L.append("Aucune cote exploitable : comparaison impossible.\n")
    else:
        L.append(f"- Bookmaker de référence : `{ev.get('reference_bookmaker')}` ; précision des horodatages : "
                 f"{ev.get('odds_snapshot_precision')}.")
        o = ev.get("overround", {})
        L.append(f"- Marge (overround) : moyenne {_pct(o.get('mean'), 2)}, médiane {_pct(o.get('median'), 2)}, "
                 f"5e–95e centiles {_pct(o.get('p05'), 2)} – {_pct(o.get('p95'), 2)}.")
        L.append(f"- Couverture en cotes : {ev.get('market_coverage')}.\n")
        rows = [[r["split"], MODEL_LABELS.get(r["model"], r["model"]), r["n"], _f(r["log_loss"]), _f(r["brier"]),
                 _f(r["roc_auc"], 3)] for r in ev.get("market_comparison_same_fights", [])]
        rows.sort(key=lambda r: (r[0] != "validation", r[1]))
        L.append(_table(rows, ["Segment", "Prédicteur", "n", "Log-loss", "Brier", "ROC-AUC"]))
        L.append("\n**Différence de log-loss modèle − marché (bootstrap par événement, IC 95 %)** — "
                 "négatif = le modèle fait mieux que le marché :\n")
        rows = [[k.split(":")[0], MODEL_LABELS.get(k.split(":")[1], k), _f(v["delta_mean"]),
                 f"[{_f(v['ci_low'])} ; {_f(v['ci_high'])}]", v["n_events"]]
                for k, v in ev.get("bootstrap_logloss_model_minus_market", {}).items()]
        L.append(_table(rows, ["Segment", "Modèle", "Δ log-loss", "IC 95 %", "Événements"]))
        L.append("\n**Régression d'« encompassing »** `logit P(A gagne) = β_marché·logit(p_marché) + β_modèle·logit(p_modèle)` "
                 "(sans constante). β_modèle > 0 significatif = information non contenue dans le prix :\n")
        rows = [[k.split(":")[0], MODEL_LABELS.get(k.split(":")[1], k), v["n"], _f(v["beta_market"], 3),
                 _f(v["beta_model"], 3), _f(v["se_model"], 3), _f(v["lr_p_value"], 4)]
                for k, v in ev.get("encompassing_regression", {}).items()]
        L.append(_table(rows, ["Segment", "Modèle", "n", "β marché", "β modèle", "e.-t. β modèle", "p (LR)"]))
        cons = ev.get("odds_source_consistency", {})
        if cons:
            L.append(f"\nCohérence entre sources de cotes : {cons}")
        L.append("")

    # --- backtest
    L.append("## 5. Simulation financière (capital fictif)\n")
    if bt.get("status") != "ok":
        L.append(bt.get("message", "Backtest non exécuté.") + "\n")
    else:
        L.append(f"- Règle de décision : {bt.get('decision_rule')} ; précision des cotes : {bt.get('odds_snapshot_precision')}.")
        p = bt.get("params", {})
        L.append(f"- Capital initial {p.get('initial_bankroll')} u. ; seuil de rendement espéré {p.get('ev_threshold')} ; "
                 f"mise max {p.get('max_stake_fraction')} du capital par combat ; exposition max "
                 f"{p.get('max_event_exposure')} par événement ; règlement {p.get('settlement')}.\n")
        for var, title in (("primary", f"Modèle seul ({MODEL_LABELS.get(bt.get('betting_model'))}) — analyse principale"),
                           ("blend", "Mélange marché + modèle — analyse secondaire")):
            if var not in bt:
                continue
            L.append(f"### {title}\n")
            if var == "blend":
                L.append("> Analyse ajoutée **après** avoir observé les résultats de l'analyse principale sur le test : "
                         "même si ses coefficients sont estimés uniquement sur des périodes antérieures, ses "
                         "résultats de test ne constituent pas une évaluation pré-enregistrée.\n")
            rows = []
            for split in ("validation", "test"):
                r = bt[var].get(split, {})
                for s, v in r.get("strategies", {}).items():
                    rows.append([split, STRAT_LABELS.get(s, s), v["n_bets"], _f(v["total_staked"], 1), _f(v["net_profit"], 1),
                                 _pct(v["roi"], 2), _f(v["final_bankroll"], 1), _pct(v["max_drawdown"], 1),
                                 _f(v["mean_odds"], 2), _pct(v["mean_estimated_ev"], 1)])
            L.append(_table(rows, ["Segment", "Stratégie", "Paris", "Misé", "Profit net", "ROI", "Capital final",
                                   "Drawdown max", "Cote moy.", "Rend. espéré estimé"]))
            L.append("")
            for split in ("validation", "test"):
                r = bt[var].get(split, {})
                b = r.get("bootstrap_unit_roi")
                if b and b.get("n_bets"):
                    L.append(f"- {split} : ROI des paris à 1 unité {_pct(b['roi_unit'], 2)}, IC 95 % bootstrap par "
                             f"événement [{_pct(b['ci_low'], 2)} ; {_pct(b['ci_high'], 2)}], "
                             f"P(ROI ≤ 0) ≈ {_f(b['p_roi_le_0'], 3)} ({b['n_bets']} paris, {b['n_events']} événements).")
                if "flat_draw_as_loss" in r:
                    L.append(f"  - variante « nul = perte » (mise fixe) : ROI {_pct(r['flat_draw_as_loss']['roi'], 2)}.")
            L.append("")
            # results by year, fixed-stake strategy
            yrows = []
            for split in ("validation", "test"):
                flat = bt[var].get(split, {}).get("strategies", {}).get("flat", {})
                for y in flat.get("by_year", []):
                    yrows.append([split, y["period"], y["n_bets"], _f(float(y["staked"]), 1),
                                  _f(float(y["profit"]), 1), _pct(float(y["roi"]), 2)])
            if yrows:
                L.append("Résultats par année (mise fixe) :\n")
                L.append(_table(yrows, ["Segment", "Année", "Paris", "Misé", "Profit net", "ROI"]))
                L.append("")
            # sensitivity on the test period, fixed stake
            sp = art / "backtest" / f"sensitivity_{var}_test.parquet"
            if sp.exists():
                sens = pd.read_parquet(sp)
                sens = sens[sens["strategy"] == "flat"]
                if len(sens):
                    piv = sens.pivot(index="ev_threshold", columns="odds_haircut", values="roi")
                    header = ["Seuil de rendement espéré"] + [f"dégradation {h:.0%}" for h in piv.columns]
                    rows = [[f"{t:.0%}"] + [_pct(v, 2) for v in piv.loc[t]] for t in piv.index]
                    L.append("Sensibilité du ROI sur le test (mise fixe ; descriptive, aucun paramètre choisi ainsi) :\n")
                    L.append(_table(rows, header))
                    L.append("")
        if "blend_coefficients" in bt:
            L.append(f"Coefficients du mélange (β_marché, β_modèle) par période : {bt['blend_coefficients']}\n")
        L.append("Les analyses de sensibilité (seuil × dégradation des cotes) sont dans "
                 "`artifacts/backtest/sensitivity_*.parquet` et dans le dashboard ; elles sont descriptives et "
                 "n'ont servi à choisir aucun paramètre.\n")

    # --- conclusions (rule-based from numbers, no free claims)
    L.append("## 6. Lecture des résultats\n")
    L.append(_conclusions(ev, bt))
    L.append("\n## 7. Limites principales\n")
    L.append("- Heure de relevé des cotes et bookmaker non documentés par les sources publiques : les cotes peuvent "
             "être des cotes de clôture, possiblement indisponibles à l'instant de décision retenu.")
    L.append("- Les profils (taille, allonge, date de naissance) proviennent d'un instantané actuel ; seules ces "
             "variables statiques sont utilisées.")
    L.append("- Combats annulés absents des données ; noms non rapprochés exclus de la comparaison au marché.")
    L.append("- Un seul découpage test ; incertitude estimée par bootstrap par événement, sans tenir compte "
             "des changements de régime.")
    L.append("- Associations statistiques, pas d'interprétation causale.")
    out = paths.reports / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def _conclusions(ev: dict, bt: dict) -> str:
    L = []
    comp = {(r["split"], r["model"]): r for r in ev.get("market_comparison_same_fights", [])}
    boots = ev.get("bootstrap_logloss_model_minus_market", {})
    enc = ev.get("encompassing_regression", {})
    for split in ("validation", "test"):
        mk = comp.get((split, "market"))
        if not mk:
            continue
        best = min((r for (s, m), r in comp.items() if s == split and m not in ("market", "baseline")),
                   key=lambda r: r["log_loss"], default=None)
        if best:
            word = "meilleure" if best["log_loss"] < mk["log_loss"] else "moins bonne"
            L.append(f"- **{split}** : le meilleur modèle sportif ({MODEL_LABELS.get(best['model'])}, log-loss "
                     f"{_f(best['log_loss'])}) a une log-loss {word} que le marché ({_f(mk['log_loss'])}) sur les "
                     f"mêmes {mk['n']} combats.")
            b = boots.get(f"{split}:{best['model']}")
            if b:
                sig = "exclut" if (b["ci_low"] > 0 or b["ci_high"] < 0) else "contient"
                L.append(f"  - L'IC 95 % de la différence {sig} zéro ([{_f(b['ci_low'])} ; {_f(b['ci_high'])}]).")
            e = enc.get(f"{split}:{best['model']}")
            if e:
                verdict = "significatif au seuil de 5 %" if e["lr_p_value"] < 0.05 else "non significatif au seuil de 5 %"
                L.append(f"  - Encompassing : β_modèle = {_f(e['beta_model'], 3)} (p = {_f(e['lr_p_value'], 4)}), {verdict} : "
                         + ("le modèle semble contenir une information partiellement absente du prix, bien qu'il soit "
                            "moins précis seul." if e["lr_p_value"] < 0.05 and e["beta_model"] > 0 else
                            "pas d'information additionnelle démontrée."))
    if bt.get("status") == "ok":
        for var in ("primary", "blend"):
            t = bt.get(var, {}).get("test", {})
            b = t.get("bootstrap_unit_roi", {})
            if b.get("n_bets"):
                demonstrated = b["ci_low"] > 0
                L.append(f"- Backtest test ({'modèle seul' if var == 'primary' else 'mélange'}) : ROI unitaire "
                         f"{_pct(b['roi_unit'], 2)}, IC 95 % [{_pct(b['ci_low'], 2)} ; {_pct(b['ci_high'], 2)}] — "
                         + ("borne basse positive, à confirmer hors échantillon et avec des cotes horodatées."
                            if demonstrated else "**aucun avantage financier démontré**."))
    L.append("- Un rendement espéré estimé positif n'est jamais une garantie de gain : il dépend entièrement de la "
             "calibration du modèle par rapport au marché.")
    return "\n".join(L)
