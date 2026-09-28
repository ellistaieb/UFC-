# Dictionnaire des données

Toutes les tables préparées sont en Parquet dans `data/processed/` (réel) ou `data/synthetic/processed/`
(démonstration synthétique). Les données brutes restent inchangées dans `data/raw/` avec un fichier
`*.provenance.json` (URL, date, SHA-256, ETag) et un `manifest.json`. **NaN = donnée absente**, jamais zéro.

## `events.parquet`
| Colonne | Description |
|---|---|
| `event_id` | Identifiant UFC Stats (fin de l'URL de l'événement) |
| `event_name`, `event_date`, `location` | Nom, date (jour), lieu |

## `fighters.parquet`
| Colonne | Description |
|---|---|
| `fighter_id` | Identifiant UFC Stats ; `unres_<nom>` si le nom n'a pas de profil ; `ambig_<nom>` si homonymes non départagés |
| `name` | Nom affiché |
| `height_cm`, `reach_cm`, `weight_lbs`, `stance`, `dob` | Profil **actuel** (seuls taille, allonge et date de naissance sont utilisés) |
| `id_status` | `ufcstats`, `unresolved` ou `ambiguous` |

## `fights.parquet`
| Colonne | Description |
|---|---|
| `fight_id`, `event_id`, `event_date`, `event_name`, `bout` | Identifiants et libellés |
| `fighter_1_*`, `fighter_2_*` | Nom, identifiant et statut de résolution dans l'ordre UFC Stats (**non aléatoire : vainqueur souvent en premier, ne jamais l'utiliser comme information**) |
| `result` | `fighter_1`, `fighter_2`, `draw`, `no_contest` |
| `method`, `end_round`, `end_time_seconds`, `time_format`, `referee` | Détails de fin de combat |
| `scheduled_rounds` | Nombre de rounds prévus (lu dans `time_format`) |
| `fight_seconds` | Durée effective ; NaN si le format est inconnu |
| `weight_class`, `is_title`, `is_women` | Catégorie |
| `has_stats` | Statistiques détaillées disponibles pour les deux combattants |
| `id_ambiguous` | Au moins un combattant non identifié de façon unique (combat exclu de la modélisation) |

## `fight_stats.parquet` (une ligne par combat × combattant, somme sur les rounds)
`kd`, `sig_landed`/`sig_att` (frappes significatives), `tot_*` (toutes frappes), `td_*` (takedowns),
`head_*`, `body_*`, `leg_*`, `distance_*`, `clinch_*`, `ground_*`, `sub_att`, `rev`, `ctrl_seconds`.
`*_landed`/`*_att` = réussies / tentées. Somme avec `min_count=1` : un combat sans aucune valeur reste NaN.

## `name_resolution.parquet`
Chaque nom rencontré dans les combats, l'identifiant attribué, le statut (`unique`, `disambiguated`,
`ambiguous`, `unresolved`) et le nombre de candidats.

## `odds.parquet` (schéma standard)
`fight_id`, `fighter_id`, `bookmaker`, `snapshot_timestamp`, `commence_timestamp`, `decimal_odds`,
`snapshot_precision` (`exact` / `date` / `unknown`), `source`. Voir `docs/odds_import_format.md`.

## `odds_reconciliation.parquet`
Une ligne par ligne source de cotes : noms, date, cotes, `match_status` (`matched`, `ambiguous`, `unmatched`),
`match_method` (`full_name`, `last_name`), `fight_id` et identifiants attribués.

## `features.parquet` (une ligne par combat, orientation canonique)
| Colonne | Description |
|---|---|
| `fighter_a_id`, `fighter_b_id` | A = plus petit identifiant (indépendant du résultat) |
| `result_a`, `y` | Résultat pour A ; `y` = 1 si A gagne, 0 s'il perd, NaN pour nul / no contest |
| `a_<f>`, `b_<f>` | Valeur de la feature pour A et B **avant** le combat |
| `diff_<f>` | `a_<f> − b_<f>` |
| `five_rounds`, `is_title`, `is_women`, `weight_limit_lbs` | Contexte |

Features `<f>` : `age_years`, `height_cm`, `reach_cm`, `n_fights`, `win_rate_shrunk`, `recent_form`,
`days_since_last` (NaN pour un débutant), `sig_landed_pm`, `sig_absorbed_pm`, `sig_acc`, `sig_def`,
`td_per15`, `td_acc`, `td_def`, `sub_att_per15`, `ctrl_share`, `elo`, `opp_elo_mean`
(+ `n_fights_with_stats`, informatif). Définitions : `docs/methodology.md`.

## Artefacts (`artifacts/`)
| Fichier | Contenu |
|---|---|
| `predictions.parquet` | `p_a` par combat, modèle, segment (`validation` / `test`) et pli (`fold`, −1 = test) ; inclut nuls et NC (`y` NaN) pour le backtest |
| `model_meta.json` | Version, dates, hyperparamètres retenus, K d'Elo, températures, modèle de pari |
| `metrics*.parquet`, `calibration.parquet`, `evaluation.json` | Scores, calibration, comparaison au marché |
| `permutation_importance.parquet` | Importance par permutation (dernière année de validation) |
| `market_probabilities.parquet` | Cotes appariées, marge et probabilité dé-margée par combat |
| `backtest/` | `bets_*`, `events_*`, `sensitivity_*`, `backtest_summary.json` |
| `final_models.pkl` | Modèles finaux (entraînés avant le test) |
