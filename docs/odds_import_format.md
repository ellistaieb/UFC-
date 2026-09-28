# Format d'import des cotes

Fichier CSV, une ligne par **combat × combattant × bookmaker × instant de relevé**. Exemple :
[`data/examples/odds_example.csv`](../data/examples/odds_example.csv) (valeurs fictives).

| Colonne | Obligatoire | Type | Description |
|---|---|---|---|
| `fight_id` | oui | texte | Identifiant UFC Stats du combat (fin de l'URL `fight-details/<id>`) |
| `fighter_id` | oui | texte | Identifiant UFC Stats du combattant (fin de l'URL `fighter-details/<id>`) |
| `bookmaker` | oui | texte | Nom du bookmaker |
| `snapshot_timestamp` | oui (vide autorisé) | ISO 8601 | Instant où la cote a été observée, idéalement avec fuseau (`2025-03-01T18:00:00Z`) |
| `commence_timestamp` | non | ISO 8601 | Heure de début du combat si connue |
| `decimal_odds` | oui | réel > 1 | Cote décimale (gain brut pour 1 unité misée) |
| `snapshot_precision` | non | `exact` / `date` / `unknown` | Déduit automatiquement : `exact` si l'horodatage est présent, sinon `unknown` |
| `source` | non | texte | Provenance |

Règles appliquées (`ufc_quant/data/odds.py`) :
- lignes avec cote ≤ 1 ou manquante rejetées (journalisé) ;
- pour chaque combat × combattant × bookmaker, on garde le relevé le plus récent **antérieur à l'instant de
  décision** (`odds.decision_hours_before_commence`, par défaut 2 h avant `commence_timestamp`, ou avant minuit
  UTC du jour de l'événement si l'heure de début est inconnue — règle conservatrice) ;
- `date` : accepté seulement si le jour de relevé précède le jour du combat, ou si
  `odds.accept_date_only_snapshots` est vrai ; `unknown` : seulement si ce paramètre est vrai ;
- la marge et la probabilité dé-margée exigent **les deux combattants, même bookmaker, même instant** ;
- aucune cote manquante n'est remplacée par une cote ultérieure.

Déclaration dans `config/config.yaml` :
```yaml
sources:
  odds:
    csv_imports: ["data/examples/odds_example.csv"]
odds:
  reference_bookmaker: "mon_bookmaker"
```

Si vos cotes ne portent que des noms, utilisez `ufc_quant.data.reconcile.match_odds_to_fights` pour obtenir
`fight_id` / `fighter_id` ; les correspondances ambiguës ou absentes sont signalées (`match_status`), jamais
validées silencieusement.

## The Odds API
`TheOddsAPIClient` (`ufc_quant/data/odds.py`) lit la clé dans la variable d'environnement `ODDS_API_KEY`
(`.env`, voir `.env.example`) et interroge l'endpoint historique v4
`/v4/historical/sports/mma_mixed_martial_arts/odds`. **Non testé** : l'hôte était bloqué dans l'environnement de
développement et la documentation actuelle n'a pas pu être relue ; l'historique nécessite un plan payant.
Revérifier la documentation avant usage. Les cotes renvoyées sont horodatées (`snapshot_precision = exact`).
