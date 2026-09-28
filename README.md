# UFC Quant

Estimation des probabilités de victoire en UFC à partir de l'information disponible **avant** chaque combat,
comparaison aux probabilités implicites des bookmakers, et simulation de gestion d'un capital **fictif**
sous contraintes de risque. Projet académique : aucun pari réel n'est effectué ni conseillé.

> **Question de recherche.** Un modèle fondé sur les informations disponibles avant un combat apporte-t-il une
> information prédictive supplémentaire par rapport aux cotes du marché, et comment cette information se
> traduit-elle dans une simulation avec contraintes de risque ?

Trois questions sont traitées séparément : (1) la qualité des probabilités prédites, (2) leur valeur ajoutée
par rapport au marché sur les mêmes combats, (3) la performance financière simulée.

## Résultats (exécution du 28/09/2026, données réelles)

Chiffres recopiés du rapport généré automatiquement (résultats par année et sensibilité inclus), [`reports/generated/report.md`](reports/generated/report.md),
qui fait foi. Test final intact : combats du 11/01/2025 au 26/09/2026.

| | Validation 2016–2024 | Test 2025–2026 |
|---|---|---|
| Log-loss, meilleur modèle sportif (tous combats décidés) | 0,6556 (GBM, n = 4 365) | 0,6310 (logistique calibrée, n = 920) |
| Log-loss Elo / référence 50 % | 0,6827 / 0,6931 | 0,6816 / 0,6931 |
| Log-loss du **marché** sur les mêmes combats (sous-ensemble avec cotes) | **0,6112** (n = 3 987) | **0,5779** (n = 608) |
| Log-loss du meilleur modèle sur ces mêmes combats | 0,6546 | 0,6190 |
| β_modèle (régression d'encompassing, GBM) | 0,225 (p = 0,003) | 0,396 (p = 0,042) |
| ROI, mises à 1 unité, modèle seul (IC 95 % bootstrap par événement) | −7,3 % [−11,9 ; −2,9] | −10,8 % [−21,4 ; +0,8] |
| ROI, mélange marché + modèle (analyse secondaire *a posteriori*) | +1,8 % [−4,9 ; +9,3] | +8,2 % [−5,4 ; +21,7] (79 paris) |

**Lecture.**
- Les modèles sportifs font nettement mieux que le hasard et qu'un Elo simple. En revanche, pris seuls, ils
  font **nettement moins bien que le marché** : l'IC 95 % de l'écart de log-loss exclut zéro sur les deux périodes.
- La régression d'encompassing trouve pourtant un coefficient positif et significatif pour le modèle : il
  semble contenir une **petite information absente du prix**, sur cet échantillon et avec ces cotes.
- Parier directement sur les probabilités du modèle **perd de l'argent**. Le modèle surestime les outsiders
  (cote moyenne jouée d'environ 3, rendement espéré estimé d'environ 40 %) : un rendement espéré estimé
  positif ne garantit aucun gain.
- Le mélange marché + modèle est positif, mais ses IC contiennent zéro : **aucun avantage financier n'est
  démontré**. Cette analyse a de plus été ajoutée après avoir vu les résultats principaux (voir limites).
- Les variables les plus associées au résultat sont l'écart d'âge (le plus jeune est favorisé), les frappes
  encaissées par minute, les takedowns, l'Elo et la force des adversaires passés. Ce sont des associations
  prédictives, pas des effets causaux.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11 testé
pip install -r requirements.txt                     # versions figées
cp .env.example .env                                # facultatif : clé The Odds API
```

## Commandes

| Étape | Commande | Rôle |
|---|---|---|
| Tout | `python -m ufc_quant all` | pipeline complet sur données réelles (~2 min) |
| Données | `python -m ufc_quant download [--refresh]` | téléchargement avec cache, délai entre requêtes, 3 tentatives, ETag |
| Préparation | `python -m ufc_quant prepare` | CSV bruts → Parquet, rapprochement des identifiants, cotes |
| Features | `python -m ufc_quant features` | choix de K (Elo) sur la validation, features chronologiques |
| Entraînement | `python -m ufc_quant train` | walk-forward, sélection, calibration, test final |
| Évaluation | `python -m ufc_quant evaluate` | scores, calibration, comparaison au marché, importances |
| Simulation | `python -m ufc_quant backtest` | 3 stratégies, sensibilité, bootstrap |
| Rapport | `python -m ufc_quant report` | `reports/generated/report.md` |
| Démo synthétique | `python -m ufc_quant all --synthetic` | tout le pipeline sur données **SYNTHÉTIQUES** (`data/synthetic/`) |
| Tests | `python -m pytest -q` | 28 tests, aussi exécutés par GitHub Actions à chaque push |
| Dashboard | `streamlit run app/streamlit_app.py` | interface (données réelles / démo synthétique) |

Un `Makefile` fournit les mêmes raccourcis (`make all`, `make demo`, `make test`, `make dashboard`).
Tous les paramètres sont centralisés dans [`config/config.yaml`](config/config.yaml).

## Données

| Source | Contenu | Accès | Remarques |
|---|---|---|---|
| [UFC Stats](http://ufcstats.com) via le miroir public [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats) | 790 événements, 8 909 combats (1994 → 26/09/2026), statistiques par round, profils | GitHub raw | ufcstats.com a répondu **403 (hôte bloqué par le proxy)** depuis l'environnement de développement ; ses conditions d'utilisation n'ont donc **pas pu être vérifiées**. À faire avant tout scraping direct. Les CSV ne sont pas redistribués ici (téléchargés localement, gitignorés). |
| [`shortlikeafox/ultimate_ufc_dataset`](https://github.com/shortlikeafox/ultimate_ufc_dataset) | cotes américaines des deux combattants, 2010 → 03/2026 | GitHub raw | **bookmaker de référence**. 6 753 combats rapprochés sur 6 924 (97,5 %) |
| [`jansen88/ufc-data`](https://github.com/jansen88/ufc-data) (betmma.tips) | cotes décimales, 2014 → 09/2023 | GitHub raw | contrôle croisé : corrélation 0,984 avec la référence sur 3 270 combats communs |
| The Odds API | cotes horodatées | clé API, plan payant pour l'historique | client implémenté mais **non testé** (hôte bloqué) |

- **Provenance** : chaque fichier brut a un `*.provenance.json` (URL, date, SHA-256, ETag) ; `manifest.json` par source.
- **Précision temporelle des cotes** : les deux sources publiques n'indiquent ni le bookmaker ni l'heure de
  relevé (`snapshot_precision = unknown`). Elles ne sont utilisées que parce que
  `odds.accept_date_only_snapshots: true` est configuré explicitement. Il peut s'agir de cotes de clôture,
  indisponibles à l'instant de décision : c'est la limite principale du backtest.
- **Rapprochement** : noms normalisés (accents, suffixes), date à ±1 jour, repli sur les noms de famille. Les
  cas ambigus ou non appariés sont conservés avec leur statut dans `odds_reconciliation.parquet` et
  `name_resolution.parquet`. Sept combats impliquant des homonymes non départagés sont exclus.
- Formats : [`docs/data_dictionary.md`](docs/data_dictionary.md), import de cotes :
  [`docs/odds_import_format.md`](docs/odds_import_format.md) (exemple : `data/examples/odds_example.csv`).

## Méthodologie (résumé)

Détails et justification : [`docs/methodology.md`](docs/methodology.md) (également affiché dans le dashboard).

- **Pas de fuite** : un moteur chronologique unique calcule chaque feature à partir des seuls combats
  strictement antérieurs à la date. Le même code sert à l'entraînement et à l'analyse « à une date » du
  dashboard. Aucune statistique de carrière actuelle n'est utilisée.
- **Orientation A/B** : UFC Stats liste le vainqueur en premier dans 64 % des cas. A est donc défini par
  l'ordre des identifiants, et les modèles sont entraînés sur le combat et son miroir, avec une prédiction
  symétrisée (P(A) + P(B) = 1 exactement).
- **Features** : un petit ensemble d'écarts A − B (âge, taille, allonge, expérience, taux de victoire lissé,
  forme, inactivité, frappes/takedowns/contrôle rétrécis vers la moyenne de population, Elo, force des
  adversaires).
- **Modèles** : référence 50 %, Elo, régression logistique L2 sans constante, gradient boosting
  (`HistGradientBoostingClassifier`). Les cotes ne sont pas des variables des modèles sportifs.
- **Validation** : walk-forward annuel 2016–2024, séparation par date (un événement n'est jamais coupé),
  test final intact depuis le 01/01/2025. Calibration par température, ajustée uniquement sur des
  prédictions antérieures.
- **Marché** : correction proportionnelle de la marge (une approximation), comparaison sur les mêmes combats,
  bootstrap par événement, régression d'encompassing.
- **Backtest** : simulation événement par événement. Les mises sont calculées sur le capital disponible avant
  l'événement, avec un plafond de 1 % par combat et de 5 % par événement. Les gains ne sont disponibles
  qu'à l'événement suivant. Stratégies : mise fixe, fraction fixe, ¼ Kelly plafonné. No contest remboursé ;
  nul remboursé par défaut, avec la variante « perte » également rapportée.

## Limites

- Horodatage et bookmaker des cotes inconnus. La couverture en cotes du test est de 66 % (la source de
  référence s'arrête en mars 2026).
- Taille, allonge et date de naissance proviennent d'un profil actuel (46 % d'allonges manquantes dans les
  profils).
- Les combats annulés sont absents des données (en pratique, ils seraient remboursés).
- Un seul test final, court (51 événements avec cotes) : incertitude large. Le bootstrap suppose des
  événements échangeables.
- Deux grilles d'hyperparamètres (K d'Elo, C de la logistique) ont été élargies après une première exécution
  où l'optimum de **validation** tombait en bord de grille. Les résultats de test de cette première exécution
  avaient été affichés ; les modèles retenus et les scores de test sont restés quasi identiques.
- L'analyse « mélange marché + modèle » a été ajoutée après avoir vu les résultats du test pour l'analyse
  principale : ce n'est pas une évaluation pré-enregistrée.
- Le plafond par événement borne les pertes mais ne modélise pas la corrélation entre combats.

## Structure

```
config/config.yaml          paramètres centralisés
ufc_quant/data/             ingestion (http, ufcstats, odds), transformation, rapprochement, synthétique
ufc_quant/features/         moteur chronologique, Elo
ufc_quant/models/           modèles symétriques, walk-forward, calibration
ufc_quant/evaluation/       découpages, métriques, comparaison au marché
ufc_quant/backtest/         finance (cotes, Kelly), moteur événementiel, analyses
ufc_quant/reporting/        rapport Markdown généré
app/streamlit_app.py        dashboard
tests/                      tests pytest (sur jeu synthétique)
docs/                       méthodologie, dictionnaire, format des cotes
```

## Démonstration synthétique

`python -m ufc_quant all --synthetic` génère un faux circuit (600 combattants fictifs « Synth Fighter NNNN »,
événements « SYNTHETIC Event NNNN ») avec des cotes horodatées, dont certaines postérieures à l'instant de
décision, afin d'exercer le filtre temporel. Les sorties vont dans `data/synthetic/`. Le dashboard et le
rapport les signalent comme **SYNTHÉTIQUES** en en-tête. Ces chiffres ne décrivent rien de réel.
