"""UFC Quant — tableau de bord Streamlit (lecture seule des artefacts du pipeline).

Lancement : streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ufc_quant.backtest.finance import devig_proportional, expected_return, fair_odds, kelly_fraction  # noqa: E402
from ufc_quant.config import get_paths, load_config  # noqa: E402
from ufc_quant.evaluation.splits import modelling_mask  # noqa: E402
from ufc_quant.features.engine import features_as_of  # noqa: E402
from ufc_quant.models.core import make_model  # noqa: E402

st.set_page_config(page_title="UFC Quant", layout="wide")

# Palette catégorielle validée pour le daltonisme (ordre fixe, jamais recyclé) + tirets/marqueurs,
# afin qu'aucune information ne repose uniquement sur la couleur.
STYLE = {
    "logistic": ("#2a78d6", "solid", "circle", "Régression logistique"),
    "logistic_cal": ("#2a78d6", "dot", "circle-open", "Logistique calibrée"),
    "gbm": ("#eb6834", "solid", "square", "Gradient boosting"),
    "gbm_cal": ("#eb6834", "dot", "square-open", "GBM calibré"),
    "elo": ("#1baf7a", "dash", "diamond", "Elo"),
    "market": ("#4a3aa7", "longdash", "triangle-up", "Marché (dé-margé)"),
    "baseline": ("#8a8983", "dashdot", "x", "Référence 50 %"),
    "blend": ("#e87ba4", "dashdot", "star", "Mélange marché + modèle"),
}
STRATS = {"flat": ("#2a78d6", "solid", "Mise fixe"), "fixed_fraction": ("#eb6834", "dash", "Fraction fixe"),
          "quarter_kelly": ("#1baf7a", "dot", "¼ Kelly plafonné")}
LAYOUT = dict(template="plotly_white", font=dict(size=13), margin=dict(l=10, r=10, t=50, b=10),
              legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))


def label(m):
    return STYLE.get(m, (None, None, None, m))[3]


@st.cache_resource
def cfg_():
    return load_config()


@st.cache_data
def load(kind: str):
    cfg = cfg_()
    paths = get_paths(cfg, synthetic=(kind == "synthetic"))
    p, a = paths.processed, paths.artifacts

    def rd(path):
        return pd.read_parquet(path) if path.exists() else None

    def rj(path):
        return json.loads(path.read_text()) if path.exists() else None

    return {
        "fights": rd(p / "fights.parquet"), "stats": rd(p / "fight_stats.parquet"),
        "fighters": rd(p / "fighters.parquet"), "features": rd(p / "features.parquet"),
        "odds": rd(p / "odds.parquet"), "preds": rd(a / "predictions.parquet"),
        "metrics": rd(a / "metrics.parquet"), "calib": rd(a / "calibration.parquet"),
        "by_year": rd(a / "metrics_by_year.parquet"), "perm": rd(a / "permutation_importance.parquet"),
        "mkt": rd(a / "market_probabilities.parquet"),
        "meta": rj(a / "model_meta.json"), "eval": rj(a / "evaluation.json"),
        "bt": rj(a / "backtest" / "backtest_summary.json"), "dmeta": rj(p / "dataset_meta.json"),
        "q_ufc": rj(p / "data_quality_ufcstats.json"), "q_odds": rj(p / "data_quality_odds.json"),
        "bt_dir": a / "backtest", "report": (paths.reports / "report.md"),
    }


# ------------------------------------------------------------------ sidebar
cfg = cfg_()
available = [k for k in ("real", "synthetic") if (get_paths(cfg, k == "synthetic").artifacts / "model_meta.json").exists()]
if not available:
    st.error("Aucun artefact trouvé. Lancer d'abord `python -m ufc_quant all` (ou `--synthetic`).")
    st.stop()
kind = st.sidebar.radio("Jeu de données", available,
                        format_func=lambda k: "Données réelles" if k == "real" else "DÉMONSTRATION SYNTHÉTIQUE")
D = load(kind)
meta, dmeta = D["meta"], D["dmeta"] or {}
if kind == "synthetic":
    st.warning("⚠️ **DÉMONSTRATION SYNTHÉTIQUE** — données générées artificiellement. "
               "Aucun chiffre affiché ne décrit l'UFC ni un marché réel.")
st.title("UFC Quant")
st.caption(f"{'Données SYNTHÉTIQUES' if kind == 'synthetic' else 'Données réelles'} · "
           f"période {dmeta.get('data_min_date')} → {dmeta.get('data_max_date')} · "
           f"modèle v{meta.get('model_version')} entraîné le {meta.get('trained_at')} · "
           f"test intact depuis le {meta.get('test_start')}")
st.sidebar.markdown("**Simulation uniquement.** Aucun pari réel n'est effectué ni conseillé.")

tabs = st.tabs(["Données", "Modèles", "Analyse d'un combat", "Simulation", "Méthodologie"])

# ------------------------------------------------------------------ Données
with tabs[0]:
    q, qo, ev = D["q_ufc"] or {}, D["q_odds"] or {}, D["eval"] or {}
    c = st.columns(4)
    c[0].metric("Combats", q.get("n_fights"))
    c[1].metric("Événements", q.get("n_events"))
    c[2].metric("Avec statistiques détaillées", q.get("fights_with_stats"))
    c[3].metric("Identités ambiguës (exclues)", q.get("fights_with_ambiguous_fighter"))
    st.subheader("Provenance")
    if kind == "real":
        st.markdown(
            "- **Combats, statistiques, profils** : tables issues de ufcstats.com via le miroir public "
            "`Greco1899/scrape_ufc_stats` (ufcstats.com inaccessible depuis l'environnement de développement).\n"
            "- **Cotes** : `ultimate_ufc_dataset` (cotes américaines, référence) et `betmma.tips` via "
            "`jansen88/ufc-data` (contrôle croisé). Bookmaker et heure de relevé non documentés.")
    else:
        st.markdown("- Générateur `ufc_quant.data.synthetic` (graine fixe). Cotes horodatées fictives.")
    st.json({"issues": q.get("result_counts"), "résolution des identifiants": q.get("id_status_counts"),
             "cotes": qo}, expanded=False)
    f = D["fights"]
    if f is not None:
        yearly = f.assign(year=f["event_date"].dt.year).groupby("year").agg(
            combats=("fight_id", "size"), avec_stats=("has_stats", "sum")).reset_index()
        if D["mkt"] is not None:
            yearly = yearly.merge(f.merge(D["mkt"][["fight_id"]], on="fight_id").assign(
                year=lambda x: x["event_date"].dt.year).groupby("year").size().rename("avec_cotes").reset_index(),
                on="year", how="left").fillna({"avec_cotes": 0})
        fig = go.Figure()
        for col, (color, dash, name) in {"combats": ("#2a78d6", "solid", "Combats"),
                                         "avec_stats": ("#eb6834", "dash", "Avec statistiques"),
                                         "avec_cotes": ("#1baf7a", "dot", "Avec cotes (réf.)")}.items():
            if col in yearly:
                fig.add_scatter(x=yearly["year"], y=yearly[col], name=name, mode="lines+markers",
                                line=dict(color=color, dash=dash, width=2), marker=dict(size=8))
        fig.update_layout(title="Couverture par année", yaxis_title="Nombre de combats", **LAYOUT)
        st.plotly_chart(fig, width="stretch")
    miss = ev.get("coverage", {}).get("feature_missing_rate_eval")
    if miss:
        st.subheader("Valeurs manquantes des features (combats depuis le début de l'entraînement)")
        st.dataframe(pd.Series(miss, name="taux manquant").sort_values(ascending=False).map("{:.1%}".format),
                     width="stretch")

# ------------------------------------------------------------------ Modèles
with tabs[1]:
    m = D["metrics"]
    if m is not None:
        split = st.radio("Segment", ["validation", "test"], horizontal=True,
                         format_func=lambda s: "Validation walk-forward" if s == "validation" else "Test final (intact)")
        t = m[m["split"] == split].copy()
        t["modèle"] = t["model"].map(label)
        st.dataframe(t[["modèle", "n", "log_loss", "brier", "ece", "accuracy", "roc_auc"]]
                     .sort_values("log_loss").rename(columns={"log_loss": "log-loss", "roc_auc": "ROC-AUC", "ece": "ECE"})
                     .style.format(precision=4), width="stretch", hide_index=True)
        st.caption("Scores sur les combats décidés. Log-loss et Brier sont les critères principaux ; "
                   "référence 50 % : log-loss 0,6931, Brier 0,25.")
        ev = D["eval"] or {}
        if ev.get("market_available"):
            st.subheader("Comparaison au marché sur les mêmes combats")
            comp = pd.DataFrame(ev["market_comparison_same_fights"])
            comp = comp[comp["split"] == split].assign(prédicteur=lambda x: x["model"].map(label))
            st.dataframe(comp[["prédicteur", "n", "log_loss", "brier", "roc_auc"]].sort_values("log_loss")
                         .style.format(precision=4), width="stretch", hide_index=True)
            enc = {k.split(":")[1]: v for k, v in ev["encompassing_regression"].items() if k.startswith(split)}
            bt = {k.split(":")[1]: v for k, v in ev["bootstrap_logloss_model_minus_market"].items() if k.startswith(split)}
            rows = [{"modèle": label(k), "Δ log-loss (modèle − marché)": bt[k]["delta_mean"],
                     "IC 95 % bas": bt[k]["ci_low"], "IC 95 % haut": bt[k]["ci_high"],
                     "β modèle (encompassing)": enc[k]["beta_model"], "p (LR)": enc[k]["lr_p_value"]} for k in enc]
            st.dataframe(pd.DataFrame(rows).style.format(precision=4), width="stretch", hide_index=True)
            st.caption("Δ < 0 : le modèle bat le marché. β modèle > 0 significatif : information absente du prix "
                       "(sur cet échantillon). Associations, pas de causalité.")
        cal = D["calib"]
        if cal is not None:
            sel = st.multiselect("Modèles (calibration)", sorted(cal["model"].unique()),
                                 default=[x for x in ("logistic", "gbm", "market") if x in set(cal["model"])],
                                 format_func=label)
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                            line=dict(color="#8a8983", dash="dot", width=1))
            for mod in sel:
                g = cal[(cal["model"] == mod) & (cal["split"] == split)]
                color, dash, marker, name = STYLE.get(mod, ("#52514e", "solid", "circle", mod))
                fig.add_scatter(x=g["p_mean"], y=g["y_rate"], mode="lines+markers", name=name,
                                error_y=dict(type="data", array=1.96 * g["y_se"], thickness=1),
                                line=dict(color=color, dash=dash, width=2), marker=dict(symbol=marker, size=9),
                                customdata=g["n"], hovertemplate="p moyen %{x:.2f}<br>fréquence %{y:.2f}<br>n=%{customdata}")
            fig.update_layout(title="Courbe de calibration (barres : IC 95 % binomial)", xaxis_title="Probabilité prédite",
                              yaxis_title="Fréquence observée", **LAYOUT)
            st.plotly_chart(fig, width="stretch")
        by = D["by_year"]
        if by is not None:
            fig = go.Figure()
            for mod in [x for x in ("elo", "logistic", "gbm") if x in set(by["model"])]:
                g = by[by["model"] == mod].sort_values("year")
                color, dash, marker, name = STYLE[mod]
                fig.add_scatter(x=g["year"], y=g["log_loss"], name=name, mode="lines+markers",
                                line=dict(color=color, dash=dash, width=2), marker=dict(symbol=marker, size=8))
            fig.add_vline(x=int(meta["test_start"][:4]) - 0.5, line_dash="dot", annotation_text="début du test")
            fig.update_layout(title="Log-loss par année (hors échantillon)", yaxis_title="Log-loss", **LAYOUT)
            st.plotly_chart(fig, width="stretch")
        st.subheader("Variables importantes")
        ev = D["eval"] or {}
        c1, c2 = st.columns(2)
        coefs = pd.Series(ev.get("logistic_coefficients_std", {})).sort_values()
        if len(coefs):
            fig = go.Figure(go.Bar(x=coefs.values, y=coefs.index, orientation="h", marker_color="#2a78d6"))
            fig.update_layout(title="Logistique : coefficients (log-odds par écart-type)", height=520, **LAYOUT)
            c1.plotly_chart(fig, width="stretch")
        perm = D["perm"]
        if perm is not None:
            g = perm[perm["model"] == "gbm"].sort_values("importance_mean")
            fig = go.Figure(go.Bar(x=g["importance_mean"], y=g["feature"], orientation="h", marker_color="#eb6834",
                                   error_x=dict(type="data", array=g["importance_std"])))
            fig.update_layout(title=f"GBM : importance par permutation (Δ log-loss, année {int(g['eval_year'].iloc[0])})",
                              height=520, **LAYOUT)
            c2.plotly_chart(fig, width="stretch")
        st.caption("Features « diff_* » = combattant A − combattant B. Les importances décrivent des associations "
                   "prédictives, pas des effets causaux.")

# ------------------------------------------------------------------ Analyse d'un combat
@st.cache_data(show_spinner="Réentraînement sur les combats antérieurs à la date d'analyse…")
def model_as_of(kind: str, date: str, family: str):
    d = load(kind)
    feats = d["features"]
    tr = feats[modelling_mask(feats) & (feats["event_date"] >= cfg["splits"]["train_start"])
               & (feats["event_date"] < pd.Timestamp(date))]
    if len(tr) < 200:
        return None, len(tr)
    return make_model(family, d["meta"]["selection"][family]["params"], cfg["project"]["random_seed"]).fit(tr), len(tr)


with tabs[2]:
    st.info("**Simulation** : la probabilité est calculée uniquement avec les combats antérieurs à la date "
            "d'analyse, par un modèle réentraîné sur ces seuls combats (hyperparamètres issus de la validation).")
    fr, fights = D["fighters"], D["fights"]
    active = pd.unique(pd.concat([fights["fighter_1_id"], fights["fighter_2_id"]]))
    fr = fr[fr["fighter_id"].isin(active) & (fr["id_status"] == "ufcstats")].sort_values("name")
    names = dict(zip(fr["fighter_id"], fr["name"]))
    ids = list(names)
    c1, c2, c3 = st.columns(3)
    a_id = c1.selectbox("Combattant A", ids, index=0, format_func=names.get)
    b_id = c2.selectbox("Combattant B", ids, index=min(1, len(ids) - 1), format_func=names.get)
    date = c3.date_input("Date d'analyse", value=pd.Timestamp(dmeta.get("data_max_date")).date() + pd.Timedelta(days=1))
    c4, c5, c6 = st.columns(3)
    rounds = c4.selectbox("Rounds prévus", [3, 5])
    title = c5.checkbox("Combat pour le titre")
    wc = c6.selectbox("Catégorie", ["Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
                                    "Middleweight", "Light Heavyweight", "Heavyweight", "Women's Strawweight",
                                    "Women's Flyweight", "Women's Bantamweight"])
    family = st.radio("Modèle", ["logistic", "gbm", "elo"], horizontal=True, format_func=label)
    if a_id == b_id:
        st.warning("Choisir deux combattants différents.")
    else:
        feats_q = features_as_of(fights, D["stats"], D["fighters"], cfg, a_id, b_id, pd.Timestamp(date),
                                 elo_k=dmeta.get("elo_k"), scheduled_rounds=rounds, is_title=title,
                                 is_women="Women" in wc, weight_class=wc)
        model, n_tr = model_as_of(kind, str(date), family)
        if model is None:
            st.error(f"Trop peu de combats antérieurs à cette date pour entraîner le modèle ({n_tr}).")
        else:
            p = float(model.predict_proba_a(pd.DataFrame([feats_q]))[0])
            k = st.columns(4)
            k[0].metric(f"P({names[a_id]} gagne)", f"{p:.1%}")
            k[1].metric(f"P({names[b_id]} gagne)", f"{1 - p:.1%}")
            k[2].metric(f"Cote juste {names[a_id]}", f"{float(fair_odds(p)):.2f}")
            k[3].metric(f"Cote juste {names[b_id]}", f"{float(fair_odds(1 - p)):.2f}")
            st.caption(f"Modèle entraîné sur {n_tr} combats antérieurs au {date}. Combats antérieurs observés : "
                       f"A = {feats_q['a_n_prior']}, B = {feats_q['b_n_prior']}.")
            if min(feats_q["a_n_prior"], feats_q["b_n_prior"]) == 0:
                st.warning("Au moins un combattant n'a aucun combat observé avant cette date : ses statistiques "
                           "sont ramenées aux moyennes de population (forte incertitude).")
            st.subheader("Comparaison à des cotes")
            real = fights[(fights["event_date"] == pd.Timestamp(date))
                          & (((fights["fighter_1_id"] == a_id) & (fights["fighter_2_id"] == b_id))
                             | ((fights["fighter_1_id"] == b_id) & (fights["fighter_2_id"] == a_id)))]
            oa0, ob0 = 2.0, 2.0
            if len(real) and D["mkt"] is not None:
                mk = D["mkt"][D["mkt"]["fight_id"] == real["fight_id"].iloc[0]]
                feats_row = D["features"].set_index("fight_id").loc[real["fight_id"].iloc[0]]
                if len(mk):
                    same = feats_row["fighter_a_id"] == a_id
                    oa0 = float(mk["odds_a"].iloc[0] if same else mk["odds_b"].iloc[0])
                    ob0 = float(mk["odds_b"].iloc[0] if same else mk["odds_a"].iloc[0])
                    st.caption(f"Cotes enregistrées pour ce combat (précision : {mk['snapshot_precision'].iloc[0]}).")
            c1, c2 = st.columns(2)
            oa = c1.number_input(f"Cote décimale {names[a_id]}", min_value=1.01, value=oa0, step=0.01)
            ob = c2.number_input(f"Cote décimale {names[b_id]}", min_value=1.01, value=ob0, step=0.01)
            pa_m, pb_m = devig_proportional(oa, ob)
            tbl = pd.DataFrame({
                "": [names[a_id], names[b_id]], "cote": [oa, ob],
                "proba implicite brute": [1 / oa, 1 / ob], "proba marché dé-margée": [pa_m, pb_m],
                "proba modèle": [p, 1 - p], "écart modèle − marché": [p - pa_m, (1 - p) - pb_m],
                "rendement espéré p·o−1": [float(expected_return(p, oa)), float(expected_return(1 - p, ob))],
                "Kelly complet": [float(kelly_fraction(p, oa)), float(kelly_fraction(1 - p, ob))],
            })
            st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if c != ""}), hide_index=True,
                         width="stretch")
            st.caption(f"Marge du bookmaker : {1 / oa + 1 / ob - 1:.2%}. La correction proportionnelle est une "
                       "approximation. Un rendement espéré positif n'est pas une garantie de gain.")
        with st.expander("Features utilisées (A − B)"):
            st.dataframe(pd.Series({k: v for k, v in feats_q.items() if k.startswith("diff_")}, name="valeur"),
                         width="stretch")

# ------------------------------------------------------------------ Simulation
with tabs[3]:
    bt = D["bt"]
    if not bt:
        st.info("Backtest non exécuté.")
    elif bt.get("status") != "ok":
        st.error(bt.get("message"))
    else:
        variant = st.radio("Probabilités utilisées", ["primary", "blend"], horizontal=True,
                           format_func=lambda v: f"Modèle seul ({label(bt['betting_model'])}) — principal"
                           if v == "primary" else "Mélange marché + modèle — secondaire")
        if variant == "blend":
            st.caption("Analyse secondaire ajoutée après observation des résultats principaux ; coefficients "
                       "estimés sur les périodes antérieures uniquement.")
        split = st.radio("Période", ["validation", "test"], horizontal=True, key="bt_split",
                         format_func=lambda s: "Validation walk-forward" if s == "validation" else "Test final")
        r = bt[variant][split]
        rows = [{"stratégie": STRATS[s][2], "paris": v["n_bets"], "misé": v["total_staked"], "profit net": v["net_profit"],
                 "ROI": v["roi"], "capital final": v["final_bankroll"], "drawdown max": v["max_drawdown"],
                 "cote moyenne": v["mean_odds"], "rendement espéré estimé": v["mean_estimated_ev"]}
                for s, v in r["strategies"].items()]
        st.dataframe(pd.DataFrame(rows).style.format({"misé": "{:.1f}", "profit net": "{:+.1f}", "ROI": "{:+.2%}",
                                                      "capital final": "{:.1f}", "drawdown max": "{:.1%}",
                                                      "cote moyenne": "{:.2f}", "rendement espéré estimé": "{:.1%}"}),
                     hide_index=True, width="stretch")
        b = r.get("bootstrap_unit_roi", {})
        if b.get("n_bets"):
            st.markdown(f"ROI des paris à 1 unité : **{b['roi_unit']:+.2%}**, IC 95 % (bootstrap par événement) "
                        f"[{b['ci_low']:+.2%} ; {b['ci_high']:+.2%}] sur {b['n_bets']} paris / {b['n_events']} événements.")
        by_year = pd.DataFrame(r["strategies"]["flat"].get("by_year", []))
        if len(by_year):
            st.markdown("**Résultats par année (mise fixe)**")
            st.dataframe(by_year.rename(columns={"period": "année", "n_bets": "paris", "staked": "misé",
                                                 "profit": "profit net", "roi": "ROI"})
                         .style.format({"misé": "{:.1f}", "profit net": "{:+.1f}", "ROI": "{:+.2%}"}),
                         hide_index=True, width="stretch")
        tag = variant
        fig = go.Figure()
        fig2 = go.Figure()
        for s, (color, dash, name) in STRATS.items():
            path = D["bt_dir"] / f"events_{tag}_{split}_{s}.parquet"
            if not path.exists():
                continue
            e = pd.read_parquet(path)
            fig.add_scatter(x=e["event_date"], y=e["bankroll_after"], name=name, mode="lines",
                            line=dict(color=color, dash=dash, width=2))
            peak = e["bankroll_after"].cummax().clip(lower=cfg["backtest"]["initial_bankroll"])
            fig2.add_scatter(x=e["event_date"], y=-(peak - e["bankroll_after"]) / peak, name=name, mode="lines",
                             line=dict(color=color, dash=dash, width=2))
        fig.add_hline(y=cfg["backtest"]["initial_bankroll"], line_dash="dot", line_color="#8a8983")
        fig.update_layout(title="Évolution du capital fictif", yaxis_title="Unités", hovermode="x unified", **LAYOUT)
        fig2.update_layout(title="Drawdown", yaxis_tickformat=".0%", hovermode="x unified", **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(fig2, width="stretch")
        sp = D["bt_dir"] / f"sensitivity_{tag}_{split}.parquet"
        if sp.exists():
            st.subheader("Sensibilité (descriptive, n'a servi à choisir aucun paramètre)")
            sens = pd.read_parquet(sp)
            strat = st.selectbox("Stratégie", list(STRATS), format_func=lambda s: STRATS[s][2])
            piv = sens[sens["strategy"] == strat].pivot(index="ev_threshold", columns="odds_haircut", values="roi")
            piv.index = [f"seuil {x:.0%}" for x in piv.index]
            piv.columns = [f"dégradation {x:.0%}" for x in piv.columns]
            st.dataframe(piv.style.format("{:+.2%}"), width="stretch")
        st.subheader("Tableau des mises")
        strat_t = st.selectbox("Stratégie ", list(STRATS), format_func=lambda s: STRATS[s][2], key="tbl")
        bp = D["bt_dir"] / f"bets_{tag}_{split}_{strat_t}.parquet"
        if bp.exists():
            bets = pd.read_parquet(bp)
            fn = D["fights"].set_index("fight_id")
            bets["combat"] = bets["fight_id"].map(fn["bout"])
            bets["résultat"] = bets["outcome"].map({"win": "✔ gagné", "loss": "✘ perdu", "refund": "↺ remboursé"})
            st.dataframe(bets[["event_date", "combat", "side", "p_side", "odds", "ev", "stake", "résultat", "pnl"]]
                         .sort_values("event_date", ascending=False)
                         .style.format({"p_side": "{:.3f}", "odds": "{:.2f}", "ev": "{:+.1%}", "stake": "{:.2f}",
                                        "pnl": "{:+.2f}"}), hide_index=True, width="stretch")
            st.caption("side : A/B selon l'orientation canonique (identifiants triés), sans lien avec le résultat.")

# ------------------------------------------------------------------ Méthodologie
with tabs[4]:
    st.markdown((ROOT / "docs" / "methodology.md").read_text(encoding="utf-8"))
    if D["report"].exists():
        with st.expander("Rapport généré (à partir des résultats calculés)"):
            st.markdown(D["report"].read_text(encoding="utf-8"))
