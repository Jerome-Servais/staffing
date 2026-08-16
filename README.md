# 📞 Prévision du volume d'appels & recommandation de staffing

Prototype Streamlit pour prévoir le nombre d'appels entrants par date et
tranche horaire, valider l'hypothèse de Poisson, comparer plusieurs modèles
de prévision (dont un modèle bayésien) et en déduire un nombre d'agents
recommandé.

## Sommaire

- [Installation](#installation)
- [Lancement](#lancement)
- [Format du fichier CSV attendu](#format-du-fichier-csv-attendu)
- [Les 5 pages de l'application](#les-5-pages-de-lapplication)
- [Architecture du code](#architecture-du-code)
- [Les modèles de prévision](#les-modèles-de-prévision)
- [Comment interpréter les métriques](#comment-interpréter-les-métriques)
- [La formule de staffing](#la-formule-de-staffing)
- [Limites connues et pistes d'amélioration](#limites-connues-et-pistes-damélioration)
- [Réponses aux questions posées](#réponses-aux-questions-posées)

---

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

> **Note Prophet/PyMC** : ces deux librairies compilent un backend au premier
> usage (cmdstanpy pour Prophet, PyTensor pour PyMC). La première exécution
> de l'application peut donc être un peu plus lente que les suivantes.

## Lancement

```bash
streamlit run app.py
```

Au premier lancement, aucune donnée n'est chargée. Deux options :

1. **Importer un fichier CSV** (vos données réelles) via la barre latérale.
2. **Aller sur la page "Données synthétiques"**, cliquer sur "🎲 Générer les
   données", puis sur "✅ Utiliser ces données dans l'application" — pratique
   pour explorer l'outil sans données réelles.

Un exemple de fichier est fourni : **`exemple_donnees_appels.csv`**
(générée avec `data_generator.py`, ~10 500 appels sur 3 mois, 5 agents).

## Format du fichier CSV attendu

| Colonne | Description |
|---|---|
| `id_appel` | Identifiant unique de l'appel |
| `date_heure` | Date et heure de l'appel (ex : `2026-01-15 09:23:11`) |
| `statut` | `repondu` ou `non_repondu` |
| `post_appel` | Temps de post-appel, en secondes |
| `temps_communication` | Durée de la conversation, en secondes |
| `temps_attente` | Temps d'attente avant prise d'appel, en secondes |
| `id_agent` | Identifiant de l'agent (vide si `non_repondu`) |

Si des colonnes sont manquantes ou si des dates sont illisibles,
l'application vous l'indique précisément dans la barre latérale (sans
planter) et vous invite à corriger le fichier.

## Les 5 pages de l'application

1. **Exploration des données** — heatmap du volume par date/tranche, courbe
   de DMT par agent, attente moyenne par tranche, taux de décroché par jour,
   table détaillée exportable.
2. **Validation de Poisson** — explications accessibles, comparaison
   histogramme observé vs courbe de Poisson théorique, test du Chi², tableau
   moyenne/variance par jour et tranche.
3. **Prédiction et comparaison** — sélection des modèles, découpage
   entraînement/test, tableau de métriques, courbe prédiction vs réel avec
   intervalle de confiance pour les modèles probabilistes.
4. **Recommandation de staffing** — choix du modèle et de l'horizon de
   prévision, sliders de seuils de qualité de service, tableau de
   recommandation, heatmap des agents recommandés, export CSV.
5. **Données synthétiques** — génère un jeu de données réaliste pour tester
   l'application, avec export CSV et chargement direct dans l'app.

## Architecture du code

```
app.py            # Application Streamlit (orchestration des 5 pages)
models.py         # Les 6 modèles de prévision + métriques + staffing
utils.py          # Prétraitement, agrégation, validation de Poisson
data_generator.py # Génération de données synthétiques réalistes
requirements.txt
exemple_donnees_appels.csv
```

### Ajouter un nouveau modèle (conçu pour être simple)

Tous les modèles héritent de `BaseForecastModel` (dans `models.py`) et
implémentent seulement deux méthodes :

```python
class MonNouveauModele(BaseForecastModel):
    name = "Mon nouveau modèle"
    is_bayesian = False
    description = "..."

    def _fit(self, train_df):
        # entraîner le modèle sur train_df (colonnes : date, heure_debut,
        # jour_semaine, jour_semaine_num, volume, ...)
        ...

    def _predict(self, future_df, extra_history=None):
        # retourner un ForecastResult(pred_mean, pred_lower, pred_upper)
        ...
```

Puis l'ajouter au dictionnaire `MODEL_REGISTRY` en bas de `models.py` :

```python
MODEL_REGISTRY["Mon nouveau modèle"] = MonNouveauModele
```

**C'est tout** : `app.py` n'a rien à modifier, le nouveau modèle apparaît
automatiquement dans les pages "Prédiction" et "Staffing".

## Les modèles de prévision

| Modèle | Type | Avantages | Inconvénients |
|---|---|---|---|
| **Régression linéaire** | Classique | Rapide, très interprétable | Peut prédire des valeurs négatives, relation supposée linéaire |
| **SARIMA** *(par tranche)* | Classique, séries temporelles | Capture bien tendance + saisonnalité hebdo si l'historique est long | Un modèle par tranche horaire ; plus lent ; sensible à un historique court |
| **Prophet** *(par tranche)* | Classique, séries temporelles | Robuste aux données manquantes, intervalle natif | Un modèle par tranche horaire ; peut sur-ajuster sur historique court (voir note ci-dessous) |
| **XGBoost (objectif Poisson)** | Machine learning | Capture des interactions non linéaires, bonnes performances | Moins interprétable, intervalle approximatif |
| **Régression de Poisson (GLM)** | Probabiliste, fréquentiste | Adaptée aux comptages, rapide, intervalle de confiance natif | L'intervalle ne porte que sur la moyenne, pas sur la variabilité réelle du comptage |
| **Poisson bayésien (PyMC)** | Probabiliste, bayésien | Intervalle de **prédiction** complet (paramètres + variabilité du comptage) ; répond à "quelle est la probabilité que le volume soit dans telle fourchette ?" | Le plus coûteux en temps de calcul (échantillonnage MCMC, ~10-20 secondes) |

> **Note technique** — SARIMA et Prophet sont conçus pour une **série
> temporelle unique** (une valeur par pas de temps). Comme nos données ont
> plusieurs tranches horaires par jour, un modèle séparé est entraîné **par
> tranche horaire**, sur la série quotidienne de cette tranche (saisonnalité
> hebdomadaire, période = 7 jours). Les modèles tabulaires (régression
> linéaire, XGBoost, GLM Poisson, Poisson bayésien), eux, utilisent une seule
> matrice de features avec le jour de la semaine et la tranche horaire en
> indicatrices (one-hot), ce qui leur permet de modéliser toutes les tranches
> à la fois.

### Le modèle bayésien, en clair

Un modèle bayésien ne donne pas une seule prédiction : il donne une
**distribution de prédictions possibles**. Concrètement :

- Le modèle est entraîné en estimant, pour chaque paramètre, non pas une
  valeur unique mais une distribution de valeurs plausibles compte tenu des
  données observées (échantillonnage MCMC, ici via PyMC/NUTS).
- Pour prédire le volume d'une tranche future, on combine ces distributions
  de paramètres avec la variabilité naturelle du comptage (loi de Poisson)
  pour obtenir, par exemple : *"on prévoit 50 appels, avec 95% de chances
  que le volume réel soit entre 40 et 60"*.
- Plus l'historique est riche et stable, plus cet intervalle se resserre.

### Exemple de code : régression de Poisson bayésienne (extrait de `models.py`)

```python
import pymc as pm

with pm.Model() as model:
    intercept = pm.Normal("intercept", mu=np.log(max(y.mean(), 1)), sigma=1.0)
    betas = pm.Normal("betas", mu=0.0, sigma=0.5, shape=n_features)
    eta = intercept + pm.math.dot(X_scaled, betas)
    mu = pm.math.exp(eta)
    pm.Poisson("y_obs", mu=mu, observed=y)

    trace = pm.sample(draws=600, tune=600, chains=2, target_accept=0.9)
```

Pour la prédiction, on calcule `mu` pour chaque tirage postérieur des
coefficients, puis on tire un comptage de Poisson pour chacun — cela donne
la distribution prédictive complète (voir `BayesianPoissonModel._predict`
dans `models.py`).

## Comment interpréter les métriques

- **RMSE / MAE** : erreur moyenne, en nombre d'appels (plus petit = mieux).
- **MAPE (%)** : erreur moyenne en pourcentage du volume réel.
- **R²** : part de la variation du volume expliquée par le modèle (proche
  de 1 = mieux).
- **Couverture IC95% (%)** : pourcentage des vraies valeurs effectivement
  comprises dans l'intervalle prédit à 95% — on vise ~95%.
- **Largeur moyenne IC** : largeur moyenne de l'intervalle ; à couverture
  égale, plus étroit = plus informatif.

## La formule de staffing

```
agents_base = CEIL( (volume_prédit × DMT_moyenne) / durée_tranche )
```

Puis on augmente progressivement le nombre d'agents tant que :

```
attente_estimée(N)  = attente_historique × (agents_base / N)  >  seuil_attente
   OU
décroché_estimé(N)  = décroché_historique × (agents_base / N)  >  seuil_décroché
```

jusqu'à respecter les deux seuils ou atteindre le maximum d'agents
disponibles (un avertissement est alors affiché).

> Cette approximation (linéaire) est volontairement simple. Une
> modélisation plus rigoureuse de l'effet du nombre d'agents sur l'attente
> utiliserait la **loi d'Erlang C** (théorie des files d'attente), qui sort
> du cadre de ce prototype mais constitue une piste d'amélioration naturelle.

## Limites connues et pistes d'amélioration

- Le staffing utilise une approximation linéaire plutôt qu'un modèle de
  file d'attente (Erlang C).
- SARIMA/Prophet par tranche horaire suppose un historique journalier
  suffisant par tranche (idéalement plusieurs mois) ; en deçà, l'application
  retombe sur une moyenne globale pour les tranches sans modèle valide.
- Le modèle bayésien est sensible au nombre d'itérations MCMC (`draws`,
  `tune` dans `BayesianPoissonModel.__init__`) : les valeurs par défaut
  privilégient la rapidité (quelques secondes) sur la précision parfaite des
  diagnostics de convergence (rhat). Pour un usage en production, augmenter
  `draws`/`tune`/`chains`.
- La DMT et la qualité de service futures sont estimées par moyenne
  historique (par jour de semaine + tranche), pas par un modèle prédictif
  dédié — cohérent avec la demande initiale, mais pourrait être affiné.

---

## Réponses aux questions posées

**1. Quels modèles proposes-tu, avantages/inconvénients ?**
Voir le tableau dans [Les modèles de prévision](#les-modèles-de-prévision).

**2. Comment implémenter le test de Poisson de manière claire et visuelle ?**
Page "Validation de Poisson" : histogramme observé superposé à la courbe
théorique de Poisson(λ), tableau moyenne/variance, test du Chi² avec
message en français (`utils.poisson_chi2_test`, `utils.poisson_validation_report`).

**3. Comment calculer le taux de décroché ?**
`utils.aggregate_by_slot` : un appel est compté "décroché" (non satisfaisant)
s'il est `non_repondu`, OU `repondu` avec un `temps_attente` supérieur au
seuil choisi dans la barre latérale.

**4. Exemples de code (Poisson bayésien + intervalles de confiance) ?**
Voir [Le modèle bayésien, en clair](#le-modèle-bayésien-en-clair) ci-dessus
et la classe `BayesianPoissonModel` dans `models.py`.

**5. Comment structurer le code pour une extension modulaire ?**
Voir [Ajouter un nouveau modèle](#ajouter-un-nouveau-modèle-conçu-pour-être-simple) :
interface commune `BaseForecastModel` + dictionnaire `MODEL_REGISTRY`.

**6. Comment générer des données synthétiques réalistes ?**
`data_generator.generate_synthetic_calls` : volume via processus de Poisson
(profils jour/heure configurables), attente via loi exponentielle, DMT par
agent avec compétence individuelle + courbe d'apprentissage progressive.

---

*Prototype développé avec Streamlit, pandas, scikit-learn, statsmodels,
Prophet, XGBoost et PyMC.*
