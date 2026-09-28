# UFC Quant — rapport généré automatiquement

- Généré le : 2026-09-28T17:59:48+00:00
- Version du modèle : 0.1.0 — entraîné le 2026-09-28T11:29:02+00:00
- Données : 1994-03-11 → 2026-09-26 (réelles)
- Tous les chiffres ci-dessous proviennent des fichiers `artifacts/` produits par le pipeline.

## 1. Données

- Combats : 8909 sur 790 événements (1994-03-11 → 2026-09-26).
- Issues : {'fighter_1': 5602, 'fighter_2': 3152, 'no_contest': 90, 'draw': 65}.
- Combats avec statistiques détaillées : 8886.
- Résolution des identifiants combattants : {'unique': 17733, 'disambiguated': 48, 'unresolved': 30, 'ambiguous': 7} ; combats exclus pour identité ambiguë : 7.
- Taux de valeurs manquantes (profils) : {'height_cm': 0.1109, 'reach_cm': 0.4617, 'dob': 0.1683, 'stance': 0.2206}.
- Cotes `ultimate_ufc_dataset` : {'rows_with_two_sided_odds': 6924, 'match_status': {'matched': 6753, 'unmatched': 170, 'ambiguous': 1}, 'matched_fights_used': 6753, 'fights_matched_by_several_rows_dropped': 0, 'date_min': '2010-03-21', 'date_max': '2026-03-28'}
- Cotes `betmma_tips` : {'rows_with_two_sided_odds': 3448, 'match_status': {'matched': 3417, 'unmatched': 31}, 'matched_fights_used': 3417, 'fights_matched_by_several_rows_dropped': 0, 'date_min': '2014-11-07', 'date_max': '2023-09-16'}
- Échantillons de modélisation : validation 4365 combats décidés, test 920 ; nuls 65, no contests 90 (exclus du score, inclus dans le backtest).

## 2. Protocole

- Entraînement à partir du 2005-01-01 ; validation walk-forward annuelle [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] ; **test final intact à partir du 2025-01-01**.
- Elo : K = 64.0 choisi sur les années de validation.
- Hyperparamètres retenus (log-loss de validation) : Référence 50 % {}; Elo {}; Régression logistique {'C': 0.003}; Gradient boosting {'learning_rate': 0.05, 'max_depth': 3, 'max_iter': 200, 'min_samples_leaf': 50, 'l2_regularization': 1.0}
- Températures de calibration (ajustées sur la validation) : {'logistic': 1.0884586946551003, 'gbm': 1.0268132316144805}
- Modèle utilisé pour le backtest (meilleure log-loss de validation) : **Gradient boosting**.

## 3. Qualité des probabilités (tous les combats décidés)

| Segment | Modèle | n | Log-loss | Brier | ECE | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|
| validation | Elo | 4365 | 0.6827 | 0.2449 | 0.016 | 0.555 | 0.580 |
| validation | GBM calibré | 4365 | 0.6560 | 0.2322 | 0.015 | 0.607 | 0.651 |
| validation | Gradient boosting | 4365 | 0.6556 | 0.2320 | 0.014 | 0.607 | 0.651 |
| validation | Logistique calibrée | 4365 | 0.6566 | 0.2324 | 0.016 | 0.611 | 0.651 |
| validation | Référence 50 % | 4365 | 0.6931 | 0.2500 | 0.007 | 0.507 | 0.500 |
| validation | Régression logistique | 4365 | 0.6566 | 0.2324 | 0.018 | 0.611 | 0.651 |
| test | Elo | 920 | 0.6816 | 0.2441 | 0.021 | 0.562 | 0.597 |
| test | GBM calibré | 920 | 0.6436 | 0.2257 | 0.035 | 0.652 | 0.684 |
| test | Gradient boosting | 920 | 0.6438 | 0.2258 | 0.039 | 0.652 | 0.684 |
| test | Logistique calibrée | 920 | 0.6310 | 0.2203 | 0.030 | 0.642 | 0.700 |
| test | Référence 50 % | 920 | 0.6931 | 0.2500 | 0.022 | 0.522 | 0.500 |
| test | Régression logistique | 920 | 0.6325 | 0.2210 | 0.034 | 0.642 | 0.700 |

Référence : log-loss d'une prédiction à 50 % = 0,6931 ; Brier = 0,25.

## 4. Valeur ajoutée par rapport au marché (mêmes combats)

- Bookmaker de référence : `ultimate_ufc_dataset_consensus` ; précision des horodatages : {'unknown': 6753}.
- Marge (overround) : moyenne 3.53 %, médiane 3.70 %, 5e–95e centiles 1.67 % – 5.35 %.
- Couverture en cotes : {'test': {'n_fights': 920, 'n_with_odds': 608, 'share': 0.6608695652173913}, 'validation': {'n_fights': 4365, 'n_with_odds': 3987, 'share': 0.9134020618556701}}.

| Segment | Prédicteur | n | Log-loss | Brier | ROC-AUC |
|---|---|---|---|---|---|
| validation | Elo | 3987 | 0.6827 | 0.2449 | 0.580 |
| validation | GBM calibré | 3987 | 0.6550 | 0.2317 | 0.653 |
| validation | Gradient boosting | 3987 | 0.6546 | 0.2315 | 0.654 |
| validation | Logistique calibrée | 3987 | 0.6557 | 0.2320 | 0.653 |
| validation | Marché (cotes dé-margées) | 3987 | 0.6112 | 0.2117 | 0.726 |
| validation | Référence 50 % | 3987 | 0.6931 | 0.2500 | 0.500 |
| validation | Régression logistique | 3987 | 0.6557 | 0.2320 | 0.653 |
| test | Elo | 608 | 0.6788 | 0.2428 | 0.602 |
| test | GBM calibré | 608 | 0.6278 | 0.2185 | 0.711 |
| test | Gradient boosting | 608 | 0.6284 | 0.2188 | 0.711 |
| test | Logistique calibrée | 608 | 0.6190 | 0.2150 | 0.719 |
| test | Marché (cotes dé-margées) | 608 | 0.5779 | 0.1973 | 0.765 |
| test | Référence 50 % | 608 | 0.6931 | 0.2500 | 0.500 |
| test | Régression logistique | 608 | 0.6215 | 0.2160 | 0.719 |

**Différence de log-loss modèle − marché (bootstrap par événement, IC 95 %)** — négatif = le modèle fait mieux que le marché :

| Segment | Modèle | Δ log-loss | IC 95 % | Événements |
|---|---|---|---|---|
| test | Elo | 0.1009 | [0.0723 ; 0.1289] | 51 |
| test | Régression logistique | 0.0436 | [0.0216 ; 0.0646] | 51 |
| test | Logistique calibrée | 0.0411 | [0.0190 ; 0.0620] | 51 |
| test | Gradient boosting | 0.0505 | [0.0287 ; 0.0730] | 51 |
| test | GBM calibré | 0.0499 | [0.0280 ; 0.0725] | 51 |
| validation | Elo | 0.0714 | [0.0616 ; 0.0809] | 370 |
| validation | Régression logistique | 0.0445 | [0.0357 ; 0.0529] | 370 |
| validation | Logistique calibrée | 0.0445 | [0.0357 ; 0.0531] | 370 |
| validation | Gradient boosting | 0.0434 | [0.0338 ; 0.0527] | 370 |
| validation | GBM calibré | 0.0438 | [0.0343 ; 0.0530] | 370 |

**Régression d'« encompassing »** `logit P(A gagne) = β_marché·logit(p_marché) + β_modèle·logit(p_modèle)` (sans constante). β_modèle > 0 significatif = information non contenue dans le prix :

| Segment | Modèle | n | β marché | β modèle | e.-t. β modèle | p (LR) |
|---|---|---|---|---|---|---|
| test | Elo | 608 | 1.181 | 0.106 | 0.255 | 0.6787 |
| test | Régression logistique | 608 | 0.979 | 0.520 | 0.215 | 0.0145 |
| test | Logistique calibrée | 608 | 0.979 | 0.478 | 0.197 | 0.0145 |
| test | Gradient boosting | 608 | 1.032 | 0.396 | 0.196 | 0.0423 |
| test | GBM calibré | 608 | 1.032 | 0.385 | 0.191 | 0.0423 |
| validation | Elo | 3987 | 1.186 | -0.106 | 0.105 | 0.3107 |
| validation | Régression logistique | 3987 | 1.087 | 0.193 | 0.083 | 0.0197 |
| validation | Logistique calibrée | 3987 | 1.089 | 0.185 | 0.081 | 0.0221 |
| validation | Gradient boosting | 3987 | 1.070 | 0.225 | 0.076 | 0.0031 |
| validation | GBM calibré | 3987 | 1.074 | 0.216 | 0.076 | 0.0047 |

Cohérence entre sources de cotes : {'betmma_tips vs ultimate_ufc_dataset_consensus': {'n_common_fights': 3270, 'corr': 0.9844497302462372, 'mean_abs_diff': 0.01845216094685616, 'share_favourite_disagreement': 0.0418960244648318}}

## 5. Simulation financière (capital fictif)

- Règle de décision : {'decision_hours_before_commence': 2.0, 'accept_date_only_snapshots': True} ; précision des cotes : {'unknown': 6753}.
- Capital initial 1000.0 u. ; seuil de rendement espéré 0.03 ; mise max 0.01 du capital par combat ; exposition max 0.05 par événement ; règlement {'no_contest': 'refund', 'draw': 'refund'}.

### Modèle seul (Gradient boosting) — analyse principale

| Segment | Stratégie | Paris | Misé | Profit net | ROI | Capital final | Drawdown max | Cote moy. | Rend. espéré estimé |
|---|---|---|---|---|---|---|---|---|---|
| validation | Mise fixe (10 u.) | 3263 | 12847.0 | -724.4 | -5.64 % | 275.6 | 75.4 % | 2.99 | 39.2 % |
| validation | Fraction fixe (0,5 %) | 3263 | 11416.7 | -698.8 | -6.12 % | 301.2 | 72.9 % | 2.99 | 39.3 % |
| validation | ¼ Kelly plafonné | 3263 | 12929.9 | -718.7 | -5.56 % | 281.3 | 75.0 % | 2.99 | 39.5 % |
| test | Mise fixe (10 u.) | 531 | 2441.1 | -242.6 | -9.94 % | 757.4 | 29.3 % | 3.40 | 48.9 % |
| test | Fraction fixe (0,5 %) | 531 | 2329.8 | -228.1 | -9.79 % | 771.9 | 28.0 % | 3.40 | 49.1 % |
| test | ¼ Kelly plafonné | 531 | 2425.2 | -254.0 | -10.47 % | 746.0 | 30.2 % | 3.40 | 49.7 % |

- validation : ROI des paris à 1 unité -7.31 %, IC 95 % bootstrap par événement [-11.94 % ; -2.86 %], P(ROI ≤ 0) ≈ 0.999 (3263 paris, 370 événements).
  - variante « nul = perte » (mise fixe) : ROI -5.64 %.
- test : ROI des paris à 1 unité -10.78 %, IC 95 % bootstrap par événement [-21.42 % ; 0.76 %], P(ROI ≤ 0) ≈ 0.965 (531 paris, 51 événements).
  - variante « nul = perte » (mise fixe) : ROI -10.77 %.

Résultats par année (mise fixe) :

| Segment | Année | Paris | Misé | Profit net | ROI |
|---|---|---|---|---|---|
| validation | 2016 | 389 | 2103.3 | 3.6 | 0.17 % |
| validation | 2017 | 360 | 1963.9 | 77.3 | 3.93 % |
| validation | 2018 | 362 | 1955.2 | -247.2 | -12.64 % |
| validation | 2019 | 380 | 1551.4 | -199.0 | -12.83 % |
| validation | 2020 | 331 | 1357.5 | 34.3 | 2.53 % |
| validation | 2021 | 383 | 1415.1 | -74.8 | -5.29 % |
| validation | 2022 | 404 | 1095.0 | -119.7 | -10.93 % |
| validation | 2023 | 343 | 788.2 | -138.0 | -17.50 % |
| validation | 2024 | 311 | 617.4 | -61.0 | -9.87 % |
| test | 2025 | 431 | 2062.1 | -107.6 | -5.22 % |
| test | 2026 | 100 | 378.9 | -135.1 | -35.64 % |

Sensibilité du ROI sur le test (mise fixe ; descriptive, aucun paramètre choisi ainsi) :

| Seuil de rendement espéré | dégradation 0% | dégradation 2% | dégradation 5% |
|---|---|---|---|
| 0% | -9.86 % | -11.27 % | -12.19 % |
| 2% | -10.28 % | -10.87 % | -12.70 % |
| 3% | -9.94 % | -11.32 % | -12.04 % |
| 5% | -9.28 % | -9.93 % | -13.62 % |
| 10% | -11.21 % | -12.93 % | -17.76 % |

### Mélange marché + modèle — analyse secondaire

> Analyse ajoutée **après** avoir observé les résultats de l'analyse principale sur le test : même si ses coefficients sont estimés uniquement sur des périodes antérieures, ses résultats de test ne constituent pas une évaluation pré-enregistrée.

| Segment | Stratégie | Paris | Misé | Profit net | ROI | Capital final | Drawdown max | Cote moy. | Rend. espéré estimé |
|---|---|---|---|---|---|---|---|---|---|
| validation | Mise fixe (10 u.) | 915 | 8760.9 | 174.1 | 1.99 % | 1174.1 | 29.4 % | 2.06 | 7.7 % |
| validation | Fraction fixe (0,5 %) | 915 | 4859.6 | 72.6 | 1.49 % | 1072.6 | 19.3 % | 2.06 | 7.8 % |
| validation | ¼ Kelly plafonné | 915 | 9298.2 | 174.1 | 1.87 % | 1174.1 | 29.5 % | 2.06 | 7.9 % |
| test | Mise fixe (10 u.) | 79 | 787.9 | 64.6 | 8.19 % | 1064.6 | 4.7 % | 1.73 | 6.2 % |
| test | Fraction fixe (0,5 %) | 79 | 399.6 | 32.6 | 8.15 % | 1032.6 | 2.4 % | 1.73 | 6.2 % |
| test | ¼ Kelly plafonné | 79 | 778.0 | 57.2 | 7.36 % | 1057.2 | 4.7 % | 1.73 | 6.2 % |

- validation : ROI des paris à 1 unité 1.83 %, IC 95 % bootstrap par événement [-4.89 % ; 9.30 %], P(ROI ≤ 0) ≈ 0.314 (915 paris, 278 événements).
  - variante « nul = perte » (mise fixe) : ROI 1.99 %.
- test : ROI des paris à 1 unité 8.21 %, IC 95 % bootstrap par événement [-5.37 % ; 21.71 %], P(ROI ≤ 0) ≈ 0.123 (79 paris, 39 événements).
  - variante « nul = perte » (mise fixe) : ROI 8.19 %.

Résultats par année (mise fixe) :

| Segment | Année | Paris | Misé | Profit net | ROI |
|---|---|---|---|---|---|
| validation | 2017 | 186 | 1734.9 | 275.1 | 15.86 % |
| validation | 2018 | 203 | 1923.6 | -138.4 | -7.19 % |
| validation | 2019 | 167 | 1542.1 | -133.2 | -8.64 % |
| validation | 2020 | 53 | 528.5 | 81.0 | 15.33 % |
| validation | 2021 | 124 | 1235.7 | -20.7 | -1.68 % |
| validation | 2022 | 90 | 876.1 | -22.1 | -2.52 % |
| validation | 2023 | 55 | 550.0 | 36.1 | 6.57 % |
| validation | 2024 | 37 | 370.0 | 96.3 | 26.01 % |
| test | 2025 | 65 | 647.9 | 49.4 | 7.62 % |
| test | 2026 | 14 | 140.0 | 15.2 | 10.84 % |

Sensibilité du ROI sur le test (mise fixe ; descriptive, aucun paramètre choisi ainsi) :

| Seuil de rendement espéré | dégradation 0% | dégradation 2% | dégradation 5% |
|---|---|---|---|
| 0% | 4.31 % | 2.63 % | 0.02 % |
| 2% | 3.11 % | 7.01 % | 8.66 % |
| 3% | 8.19 % | 16.58 % | 8.36 % |
| 5% | 4.71 % | 12.69 % | 17.57 % |
| 10% | 26.46 % | 66.71 % | 63.65 % |

Coefficients du mélange (β_marché, β_modèle) par période : {'2017': {'beta_market': 0.9007600187627588, 'beta_model': 0.403769311623968}, '2018': {'beta_market': 0.8668160092390917, 'beta_model': 0.5564783812447656}, '2019': {'beta_market': 0.9920295398840817, 'beta_model': 0.37389860460374175}, '2020': {'beta_market': 1.017254254453376, 'beta_model': 0.2480708038289291}, '2021': {'beta_market': 0.9888441671324958, 'beta_model': 0.32813262357045925}, '2022': {'beta_market': 1.0106693247594647, 'beta_model': 0.26976899627050355}, '2023': {'beta_market': 1.0326229622043657, 'beta_model': 0.24915956258162675}, '2024': {'beta_market': 1.064853209440694, 'beta_model': 0.2312742759131534}, 'test': {'beta_market': 1.0699021282927272, 'beta_model': 0.22506239993033453}}

Les analyses de sensibilité (seuil × dégradation des cotes) sont dans `artifacts/backtest/sensitivity_*.parquet` et dans le dashboard ; elles sont descriptives et n'ont servi à choisir aucun paramètre.

## 6. Lecture des résultats

- **validation** : le meilleur modèle sportif (Gradient boosting, log-loss 0.6546) a une log-loss moins bonne que le marché (0.6112) sur les mêmes 3987 combats.
  - L'IC 95 % de la différence exclut zéro ([0.0338 ; 0.0527]).
  - Encompassing : β_modèle = 0.225 (p = 0.0031), significatif au seuil de 5 % : le modèle semble contenir une information partiellement absente du prix, bien qu'il soit moins précis seul.
- **test** : le meilleur modèle sportif (Logistique calibrée, log-loss 0.6190) a une log-loss moins bonne que le marché (0.5779) sur les mêmes 608 combats.
  - L'IC 95 % de la différence exclut zéro ([0.0190 ; 0.0620]).
  - Encompassing : β_modèle = 0.478 (p = 0.0145), significatif au seuil de 5 % : le modèle semble contenir une information partiellement absente du prix, bien qu'il soit moins précis seul.
- Backtest test (modèle seul) : ROI unitaire -10.78 %, IC 95 % [-21.42 % ; 0.76 %] — **aucun avantage financier démontré**.
- Backtest test (mélange) : ROI unitaire 8.21 %, IC 95 % [-5.37 % ; 21.71 %] — **aucun avantage financier démontré**.
- Un rendement espéré estimé positif n'est jamais une garantie de gain : il dépend entièrement de la calibration du modèle par rapport au marché.

## 7. Limites principales

- Heure de relevé des cotes et bookmaker non documentés par les sources publiques : les cotes peuvent être des cotes de clôture, possiblement indisponibles à l'instant de décision retenu.
- Les profils (taille, allonge, date de naissance) proviennent d'un instantané actuel ; seules ces variables statiques sont utilisées.
- Combats annulés absents des données ; noms non rapprochés exclus de la comparaison au marché.
- Un seul découpage test ; incertitude estimée par bootstrap par événement, sans tenir compte des changements de régime.
- Associations statistiques, pas d'interprétation causale.