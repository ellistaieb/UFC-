## Méthodologie, hypothèses et limites

### Question de recherche
Un modèle fondé uniquement sur l'information disponible **avant** un combat apporte-t-il une information
prédictive **supplémentaire** par rapport aux cotes, et que devient cette information dans une simulation
de mises sous contraintes de risque ? Trois questions sont séparées :

1. **Qualité des probabilités** : log-loss (critère principal), Brier, calibration ; accuracy et ROC-AUC en complément.
2. **Valeur ajoutée face au marché**, sur exactement les mêmes combats : différence de log-loss avec IC par
   bootstrap sur les événements, et régression d'« encompassing »
   `logit P = β_m·logit(p_marché) + β_s·logit(p_modèle)` (β_s > 0 significatif = information absente du prix).
3. **Performance financière simulée** : capital fictif, aucune mise réelle.

Un résultat négatif est une conclusion acceptable.

### Prévention des fuites de données
- **Un seul moteur chronologique** (`ufc_quant/features/engine.py`) : pour chaque date, on photographie l'état
  de chaque combattant, puis seulement ensuite on intègre les résultats de la date. Tous les combats d'une
  même date (même événement) utilisent l'information strictement antérieure.
- Les statistiques de carrière actuelles de UFC Stats ne sont **jamais** utilisées. Seuls la date de
  naissance, la taille et l'allonge (quasi statiques) proviennent du profil actuel.
- Les moyennes de population servant au rétrécissement sont elles aussi calculées sur le passé uniquement.
- Imputation et normalisation sont ajustées dans le pipeline scikit-learn, sur l'échantillon d'entraînement de chaque pli.
- Test automatique : supprimer tous les combats futurs et brouiller les résultats du jour ne change aucune feature.

### Orientation A/B
UFC Stats liste presque toujours le vainqueur en premier : utiliser cet ordre serait une fuite massive. Le
combattant A est celui dont l'identifiant est le plus petit (hachage sans lien avec le résultat). Les modèles
sont entraînés sur le combat et sa version inversée (même pli temporel) et la prédiction est symétrisée :
`p = ½ [f(A,B) + 1 − f(B,A)]`, d'où `P(A bat B) + P(B bat A) = 1` exactement. Chaque combat est évalué une seule fois.

### Features (petit ensemble fiable, différences A − B)
Âge, taille, allonge ; nombre de combats observés ; taux de victoire lissé `(V + ½N + 1)/(n + 2)` ; forme
récente (3 derniers) ; jours depuis le dernier combat (manquant pour un débutant) ; frappes significatives
portées/encaissées par minute ; précision et défense en frappes ; takedowns par 15 min, précision, défense ;
tentatives de soumission ; part du temps en contrôle ; Elo ; Elo moyen des adversaires passés. Contexte
(GBM seulement) : 5 rounds, titre, féminin, limite de poids.

Pour les taux : totaux des numérateurs et dénominateurs, avec rétrécissement vers la moyenne de population
(`(x + k·prior)/(n + k)`, k = 15 minutes ou 20 tentatives). Un débutant reçoit la moyenne de population, pas zéro.
Une statistique absente reste manquante (NaN), distincte de zéro.

### Nuls et no contests
| Module | Nul | No contest |
|---|---|---|
| Features | combat compté, score 0,5 | combat compté (expérience, temps, stats), pas de résultat |
| Elo | score 0,5 | aucune mise à jour |
| Modèles (entraînement, scores) | exclu (cible binaire) | exclu |
| Backtest | remboursement par défaut ; variante « perte » rapportée | remboursement |

### Modèles et validation
Référence 50 % → Elo (K choisi en validation) → régression logistique L2 sans constante → gradient boosting
(`HistGradientBoostingClassifier`, NaN natifs). Les cotes ne sont **pas** des variables des modèles sportifs.
Validation walk-forward annuelle 2016–2024 (entraînement sur toutes les années antérieures), puis **test final
intact** à partir du 1er janvier 2025 : un seul ajustement, aucune décision prise sur le test. Calibration par
température (un paramètre, préserve la symétrie) ajustée uniquement sur des prédictions hors échantillon
antérieures à la période évaluée.

### Cotes
- Cotes décimales ; probabilité implicite `q = 1/o` ; marge `q_A + q_B − 1` sur les deux côtés d'un même
  bookmaker au même instant ; correction proportionnelle `p_i = q_i / Σq` (approximation : ignore le biais favori/outsider).
- Cote juste `1/p` ; rendement espéré `p·o − 1` ; Kelly `(p·o − 1)/(o − 1)` ramené à 0 si négatif.
- **Instant de décision** configurable (2 h avant le début). Seules les cotes antérieures sont retenues, la
  plus récente par bookmaker ; une cote manquante n'est jamais remplacée par une cote future.
- **Précision temporelle réelle** : les sources publiques utilisées ne donnent ni l'heure du relevé ni le
  bookmaker (`snapshot_precision = unknown`). Elles ne sont acceptées que parce que
  `accept_date_only_snapshots: true` est explicitement configuré, et chaque sortie le signale : il peut
  s'agir de cotes de clôture, non disponibles à l'instant de décision.

### Backtest
Événements traités chronologiquement. Les mises d'un événement sont calculées sur le capital disponible
**avant** l'événement, un seul côté par combat (celui au rendement espéré le plus élevé, s'il dépasse le seuil),
plafond de 1 % par combat puis de 5 % par événement (réduction proportionnelle). Les gains d'un événement ne sont
disponibles qu'à l'événement suivant. Stratégies : mise fixe 10 u., fraction fixe 0,5 %, ¼ Kelly plafonné.
Les paramètres sont illustratifs et fixés avant le test ; les analyses de sensibilité (seuil × dégradation des
cotes) sont descriptives. Pas de Sharpe annualisé (fréquence des rendements irrégulière).

**Limites du plafond par événement** : il borne la perte d'un événement mais ne modélise pas les corrélations
(ex. biais commun du modèle sur une catégorie) ; les combats d'une même soirée ne sont pas indépendants.

### Principaux biais et limites
- Horodatage et bookmaker des cotes inconnus (voir plus haut) ; couverture en cotes incomplète (noms non rapprochés).
- Profils issus d'un instantané actuel (taille, allonge, date de naissance ; 46 % d'allonges manquantes).
- Combats annulés absents des données ; 7 combats à identité ambiguë exclus.
- Un seul test final (2025–2026) : petit échantillon, incertitude large.
- Le bootstrap par événement suppose des événements échangeables (pas de changement de régime).
- Les importances de variables et coefficients sont des associations prédictives, pas des effets causaux.
- L'analyse « mélange marché + modèle » a été ajoutée après observation des résultats principaux sur le test.
