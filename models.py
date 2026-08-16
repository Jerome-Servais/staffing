"""
models.py
=========
Modeles de prevision du volume d'appels (par date + tranche horaire) et
moteur de recommandation de staffing.

Architecture pensee pour etre MODULAIRE : chaque modele est une classe qui
implemente la meme interface (fit / predict). Pour ajouter un nouveau
modele, il suffit de :
    1. Creer une classe qui herite de `BaseForecastModel`.
    2. Implementer `_fit(self, train_df)` et `_predict(self, future_df)`.
    3. L'ajouter au dictionnaire `MODEL_REGISTRY` en bas de ce fichier.
L'application Streamlit (app.py) n'a alors rien a modifier : elle lit
`MODEL_REGISTRY` pour peupler la liste des modeles disponibles.

Modeles fournis (5 + 1 variante probabiliste classique) :
    - Regression lineaire              (LinearRegressionModel)
    - SARIMA (par tranche horaire)     (SarimaModel)
    - Prophet (par tranche horaire)    (ProphetModel)
    - XGBoost (objectif Poisson)       (XGBoostModel)
    - Regression de Poisson (GLM)      (PoissonRegressionModel)   [probabiliste, frequentiste]
    - Regression de Poisson Bayesienne (BayesianPoissonModel)     [probabiliste, bayesien]
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Ingenierie des features tabulaires (partagee par les modeles tabulaires)
# ---------------------------------------------------------------------------

class TabularFeatureBuilder:
    """
    Construit une matrice de features tabulaires a partir du DataFrame
    agrege (sortie de `utils.aggregate_by_slot`), utilisable par les
    modeles de type regression / arbre (LinearRegression, XGBoost, GLM
    Poisson, Poisson Bayesien).

    Features generees :
        - one-hot du jour de la semaine (jour_semaine_num, 0=lundi)
        - one-hot de la tranche horaire (heure_debut)
        - tendance temporelle (nombre de jours depuis la 1ere date vue
          a l'entrainement -> capture une derive lente du volume)
        - week_end (0/1)
        - volume_lag_7 (optionnel) : volume observe 7 jours plus tot pour
          la meme tranche horaire (capture l'autocorrelation hebdomadaire)

    Le builder memorise les categories vues a l'entrainement (`fit`) afin
    que `transform` produise exactement les memes colonnes pour les
    donnees futures, meme si certaines categories n'apparaissent pas dans
    le futur (colonnes mises a 0).
    """

    def __init__(self, use_lag_7: bool = True):
        self.use_lag_7 = use_lag_7
        self.weekday_categories_ = None
        self.slot_categories_ = None
        self.date_ref_ = None
        self.history_: Optional[pd.DataFrame] = None  # pour les lags
        self.feature_names_: list = []

    def fit(self, train_df: pd.DataFrame) -> "TabularFeatureBuilder":
        self.weekday_categories_ = sorted(train_df["jour_semaine_num"].unique())
        self.slot_categories_ = sorted(train_df["heure_debut"].unique())
        self.date_ref_ = pd.to_datetime(train_df["date"]).min()
        self.history_ = train_df[["date", "heure_debut", "volume"]].copy()
        return self

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        dates = pd.to_datetime(df["date"])
        out["trend"] = (dates - self.date_ref_).dt.days.astype(float)
        out["week_end"] = df["jour_semaine_num"].isin([5, 6]).astype(int)

        for w in self.weekday_categories_:
            out[f"jour_{w}"] = (df["jour_semaine_num"] == w).astype(int)
        for s in self.slot_categories_:
            out[f"slot_{s}"] = (df["heure_debut"] == s).astype(int)
        return out

    def _lag_7(self, df: pd.DataFrame, extra_history: Optional[pd.DataFrame]) -> pd.Series:
        history = self.history_
        if extra_history is not None:
            history = pd.concat([history, extra_history], ignore_index=True)
        history = history.drop_duplicates(subset=["date", "heure_debut"], keep="last")
        history = history.rename(columns={"volume": "volume_lag_7"})
        history["date"] = pd.to_datetime(history["date"]) + pd.Timedelta(days=7)

        merged = df[["date", "heure_debut"]].copy()
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.merge(history, on=["date", "heure_debut"], how="left")
        # a defaut de lag connu (debut de periode), on retombe sur la
        # moyenne globale du slot observee a l'entrainement
        fallback = self.history_.groupby("heure_debut")["volume"].mean()
        merged["volume_lag_7"] = merged.apply(
            lambda r: r["volume_lag_7"] if pd.notna(r["volume_lag_7"])
            else fallback.get(r["heure_debut"], self.history_["volume"].mean()),
            axis=1,
        )
        return merged["volume_lag_7"].values

    def transform(self, df: pd.DataFrame, extra_history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        extra_history : DataFrame optionnel (date, heure_debut, volume)
            contenant des observations reelles supplementaires (ex : le
            jeu de test) permettant de calculer le lag-7 sans fuite de
            donnees (on n'utilise que des volumes deja "passes" au moment
            de la prevision).
        """
        feats = self._base_features(df)
        if self.use_lag_7:
            feats["volume_lag_7"] = self._lag_7(df, extra_history)
        self.feature_names_ = list(feats.columns)
        return feats


# ---------------------------------------------------------------------------
# Interface commune des modeles
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Sortie standardisee de tous les modeles : moyenne + (eventuel) intervalle a 95%."""
    pred_mean: np.ndarray
    pred_lower: Optional[np.ndarray] = None
    pred_upper: Optional[np.ndarray] = None

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"pred_mean": self.pred_mean})
        df["pred_lower"] = self.pred_lower if self.pred_lower is not None else np.nan
        df["pred_upper"] = self.pred_upper if self.pred_upper is not None else np.nan
        return df


class BaseForecastModel:
    """
    Interface commune. `train_df` et `future_df` sont des DataFrames issus
    de `utils.aggregate_by_slot` (colonnes : date, tranche, heure_debut,
    jour_semaine, jour_semaine_num, volume, ...).
    """
    name: str = "BaseModel"
    is_bayesian: bool = False
    description: str = ""

    def fit(self, train_df: pd.DataFrame) -> "BaseForecastModel":
        self._fit(train_df)
        self._fitted = True
        return self

    def predict(self, future_df: pd.DataFrame, extra_history: Optional[pd.DataFrame] = None) -> ForecastResult:
        if not getattr(self, "_fitted", False):
            raise RuntimeError(f"Le modele {self.name} doit etre entraine (fit) avant predict().")
        return self._predict(future_df, extra_history=extra_history)

    # A implementer dans chaque sous-classe
    def _fit(self, train_df: pd.DataFrame):
        raise NotImplementedError

    def _predict(self, future_df: pd.DataFrame, extra_history: Optional[pd.DataFrame] = None) -> ForecastResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Regression lineaire (baseline)
# ---------------------------------------------------------------------------

class LinearRegressionModel(BaseForecastModel):
    name = "Regression lineaire"
    is_bayesian = False
    description = (
        "Modele de reference (baseline). Rapide et tres interpretable, mais "
        "ne capture pas naturellement le caractere "
        "discret/positif des comptages d'appels (peut predire des valeurs "
        "negatives) et suppose une relation lineaire entre les features et "
        "le volume."
    )

    def __init__(self):
        self.builder = TabularFeatureBuilder(use_lag_7=True)
        self.model = LinearRegression()

    def _fit(self, train_df):
        self.builder.fit(train_df)
        X = self.builder.transform(train_df)
        y = train_df["volume"].values
        self.model.fit(X, y)
        self._residual_std = float(np.std(y - self.model.predict(X)))

    def _predict(self, future_df, extra_history=None):
        X = self.builder.transform(future_df, extra_history=extra_history)
        mean = np.clip(self.model.predict(X), 0, None)
        # Intervalle approximatif base sur l'ecart-type des residus d'entrainement
        z = stats.norm.ppf(0.975)
        lower = np.clip(mean - z * self._residual_std, 0, None)
        upper = mean + z * self._residual_std
        return ForecastResult(mean, lower, upper)


# ---------------------------------------------------------------------------
# 2. SARIMA (un modele par tranche horaire)
# ---------------------------------------------------------------------------

class SarimaModel(BaseForecastModel):
    name = "SARIMA"
    is_bayesian = False
    description = (
        "Un modele SARIMA est entraine independamment pour chaque tranche "
        "horaire, sur la serie quotidienne du volume de cette tranche, avec "
        "une saisonnalite hebdomadaire (periode = 7 jours). Bon choix si "
        "l'historique est suffisant (plusieurs mois) et que la saisonnalite "
        "hebdomadaire est stable ; plus lent a entrainer que les modeles "
        "tabulaires et moins flexible si de nouvelles tranches apparaissent."
    )

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.slot_models_ = {}
        self.slot_last_date_ = {}

    def _series_for_slot(self, df, slot):
        sub = df[df["heure_debut"] == slot][["date", "volume"]].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.groupby("date")["volume"].mean().asfreq("D")
        sub = sub.interpolate(limit_direction="both")
        return sub

    def _fit(self, train_df):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        for slot in sorted(train_df["heure_debut"].unique()):
            series = self._series_for_slot(train_df, slot)
            if len(series) < 14 or series.nunique() <= 1:
                continue  # historique insuffisant pour ce slot
            try:
                fitted = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                self.slot_models_[slot] = fitted
                self.slot_last_date_[slot] = series.index.max()
            except Exception:
                continue
        self._global_mean = train_df["volume"].mean()

    def _predict(self, future_df, extra_history=None):
        means, lowers, uppers = [], [], []
        for _, row in future_df.iterrows():
            slot = row["heure_debut"]
            target_date = pd.to_datetime(row["date"])
            model = self.slot_models_.get(slot)
            if model is None:
                means.append(self._global_mean)
                lowers.append(max(self._global_mean * 0.5, 0))
                uppers.append(self._global_mean * 1.5)
                continue
            last_date = self.slot_last_date_[slot]
            steps_ahead = (target_date - last_date).days
            if steps_ahead < 1:
                steps_ahead = 1
            forecast = model.get_forecast(steps=steps_ahead)
            ci = forecast.conf_int(alpha=0.05)
            idx = steps_ahead - 1
            mean_val = max(float(forecast.predicted_mean.iloc[idx]), 0)
            low_val = max(float(ci.iloc[idx, 0]), 0)
            up_val = max(float(ci.iloc[idx, 1]), 0)
            means.append(mean_val)
            lowers.append(low_val)
            uppers.append(up_val)
        return ForecastResult(np.array(means), np.array(lowers), np.array(uppers))


# ---------------------------------------------------------------------------
# 3. Prophet (un modele par tranche horaire)
# ---------------------------------------------------------------------------

class ProphetModel(BaseForecastModel):
    name = "Prophet"
    is_bayesian = False
    description = (
        "Comme pour SARIMA, un modele Prophet distinct est entraine par "
        "tranche horaire sur la serie quotidienne. Prophet gere bien les "
        "tendances et la saisonnalite hebdomadaire automatiquement, et "
        "fournit nativement un intervalle de confiance. Tres robuste aux "
        "donnees manquantes mais peut etre plus lent (un modele par "
        "tranche horaire)."
    )

    def __init__(self):
        self.slot_models_ = {}
        self._global_mean = 0.0

    def _fit(self, train_df):
        from prophet import Prophet

        for slot in sorted(train_df["heure_debut"].unique()):
            sub = train_df[train_df["heure_debut"] == slot][["date", "volume"]].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.groupby("date")["volume"].mean().reset_index()
            sub.columns = ["ds", "y"]
            if len(sub) < 14 or sub["y"].nunique() <= 1:
                continue
            try:
                # growth='flat' + peu de changepoints : le volume d'appels
                # d'un centre d'appels n'a generalement pas de tendance
                # forte (contrairement a une serie de vente par ex.), et
                # l'historique disponible par tranche est souvent court
                # (quelques mois). Le growth lineaire par defaut de Prophet
                # a tendance a sur-ajuster sur ce type de serie courte.
                m = Prophet(
                    growth="flat",
                    n_changepoints=4,
                    changepoint_prior_scale=0.01,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    daily_seasonality=False,
                    interval_width=0.95,
                )
                m.fit(sub)
                self.slot_models_[slot] = m
            except Exception:
                continue
        self._global_mean = train_df["volume"].mean()

    def _predict(self, future_df, extra_history=None):
        means, lowers, uppers = [], [], []
        for slot in sorted(future_df["heure_debut"].unique()):
            model = self.slot_models_.get(slot)
            mask = future_df["heure_debut"] == slot
            dates = pd.to_datetime(future_df.loc[mask, "date"])
            if model is None:
                for _ in dates:
                    means.append(self._global_mean)
                    lowers.append(max(self._global_mean * 0.5, 0))
                    uppers.append(self._global_mean * 1.5)
                continue
            future = pd.DataFrame({"ds": dates})
            fc = model.predict(future)
            means.extend(np.clip(fc["yhat"].values, 0, None))
            lowers.extend(np.clip(fc["yhat_lower"].values, 0, None))
            uppers.extend(np.clip(fc["yhat_upper"].values, 0, None))
        # Replacer les predictions (calculees slot par slot) dans l'ordre
        # d'origine des lignes de future_df.
        positions = np.concatenate([
            np.where(future_df["heure_debut"].values == s)[0]
            for s in sorted(future_df["heure_debut"].unique())
        ])
        means, lowers, uppers = np.array(means), np.array(lowers), np.array(uppers)
        result = np.empty_like(means)
        result_l = np.empty_like(lowers)
        result_u = np.empty_like(uppers)
        result[positions] = means
        result_l[positions] = lowers
        result_u[positions] = uppers
        return ForecastResult(result, result_l, result_u)


# ---------------------------------------------------------------------------
# 4. XGBoost (objectif Poisson, adapte aux comptages)
# ---------------------------------------------------------------------------

class XGBoostModel(BaseForecastModel):
    name = "XGBoost (Poisson)"
    is_bayesian = False
    description = (
        "Gradient boosting (XGBoost) avec objectif 'count:poisson', adapte "
        "aux donnees de comptage. Capture des interactions non lineaires "
        "entre jour de semaine / tranche horaire / tendance que la "
        "regression lineaire ne peut pas voir. Tres performant si "
        "l'historique est suffisant, mais moins interpretable et l'intervalle "
        "de confiance fourni ici est approximatif (base sur les residus)."
    )

    def __init__(self, n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.08):
        import xgboost as xgb

        self.builder = TabularFeatureBuilder(use_lag_7=True)
        self.model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )

    def _fit(self, train_df):
        self.builder.fit(train_df)
        X = self.builder.transform(train_df)
        y = train_df["volume"].values
        self.model.fit(X, y)
        self._residual_std = float(np.std(y - self.model.predict(X)))

    def _predict(self, future_df, extra_history=None):
        X = self.builder.transform(future_df, extra_history=extra_history)
        mean = np.clip(self.model.predict(X), 0, None)
        z = stats.norm.ppf(0.975)
        lower = np.clip(mean - z * self._residual_std, 0, None)
        upper = mean + z * self._residual_std
        return ForecastResult(mean, lower, upper)


# ---------------------------------------------------------------------------
# 5. Regression de Poisson (GLM, statsmodels) -- probabiliste frequentiste
# ---------------------------------------------------------------------------

class PoissonRegressionModel(BaseForecastModel):
    name = "Regression de Poisson (GLM)"
    is_bayesian = False
    description = (
        "Modele lineaire generalise (GLM) avec une loi de Poisson : adapte "
        "nativement aux comptages (predictions toujours positives, variance "
        "liee a la moyenne). Fournit un intervalle de confiance a 95% sur "
        "la moyenne predite. Plus simple et plus rapide que la version "
        "bayesienne, mais l'incertitude qu'il fournit ne porte que sur le "
        "parametre moyen, pas sur la variabilite intrinseque du comptage "
        "(contrairement a l'intervalle predictif bayesien)."
    )

    def __init__(self):
        self.builder = TabularFeatureBuilder(use_lag_7=True)
        self.result_ = None

    def _fit(self, train_df):
        import statsmodels.api as sm

        self.builder.fit(train_df)
        X = self.builder.transform(train_df)
        X_const = sm.add_constant(X, has_constant="add")
        y = train_df["volume"].values
        model = sm.GLM(y, X_const, family=sm.families.Poisson())
        self.result_ = model.fit()
        self._X_columns = X_const.columns

    def _predict(self, future_df, extra_history=None):
        import statsmodels.api as sm

        X = self.builder.transform(future_df, extra_history=extra_history)
        X_const = sm.add_constant(X, has_constant="add")
        X_const = X_const.reindex(columns=self._X_columns, fill_value=0)
        pred = self.result_.get_prediction(X_const)
        summary = pred.summary_frame(alpha=0.05)
        mean = summary["mean"].values
        lower = summary["mean_ci_lower"].values
        upper = summary["mean_ci_upper"].values
        return ForecastResult(mean, np.clip(lower, 0, None), upper)


# ---------------------------------------------------------------------------
# 6. Regression de Poisson Bayesienne (PyMC) -- incertitude complete
# ---------------------------------------------------------------------------

class BayesianPoissonModel(BaseForecastModel):
    name = "Poisson Bayesien (PyMC)"
    is_bayesian = True
    description = (
        "Regression de Poisson bayesienne (PyMC). Au lieu d'une seule valeur "
        "predite, on obtient une distribution complete de previsions "
        "possibles : la moyenne, mais aussi un intervalle credible a 95% "
        "qui integre a la fois l'incertitude sur les parametres du modele "
        "et la variabilite naturelle du nombre d'appels (loi de Poisson). "
        "C'est le seul modele ici qui repond directement a la question "
        "'quelle est la probabilite que le volume soit dans telle "
        "fourchette ?'. Plus couteux en temps de calcul (echantillonnage "
        "MCMC) que les autres modeles."
    )

    def __init__(self, draws: int = 600, tune: int = 600, chains: int = 2, seed: int = 42):
        self.builder = TabularFeatureBuilder(use_lag_7=True)
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.seed = seed
        self.coef_samples_ = None  # (n_samples, n_features+1)
        self.feature_columns_ = None

    def _fit(self, train_df):
        import pymc as pm

        self.builder.fit(train_df)
        X = self.builder.transform(train_df)
        self.feature_columns_ = list(X.columns)
        X_np = X.values.astype(float)
        # Standardisation des features continues (trend, lag) pour faciliter
        # l'echantillonnage MCMC ; les indicatrices 0/1 sont laissees telles
        # quelles (la standardisation casserait leur interpretation directe).
        self._feature_mean = X_np.mean(axis=0)
        self._feature_std = X_np.std(axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        is_binary = np.array([set(np.unique(X_np[:, j])) <= {0.0, 1.0} for j in range(X_np.shape[1])])
        self._feature_mean[is_binary] = 0.0
        self._feature_std[is_binary] = 1.0
        X_scaled = (X_np - self._feature_mean) / self._feature_std

        y = train_df["volume"].values.astype(int)
        n_features = X_scaled.shape[1]

        with pm.Model() as model:
            intercept = pm.Normal("intercept", mu=np.log(max(y.mean(), 1)), sigma=1.0)
            betas = pm.Normal("betas", mu=0.0, sigma=0.5, shape=n_features)
            eta = intercept + pm.math.dot(X_scaled, betas)
            mu = pm.math.exp(eta)
            pm.Poisson("y_obs", mu=mu, observed=y)

            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=1,
                random_seed=self.seed,
                progressbar=False,
                target_accept=0.9,
            )

        post = trace.posterior
        intercept_samples = post["intercept"].values.reshape(-1)
        beta_samples = post["betas"].values.reshape(-1, n_features)
        self.coef_samples_ = np.concatenate(
            [intercept_samples.reshape(-1, 1), beta_samples], axis=1
        )  # (n_samples, 1 + n_features)

    def _predict(self, future_df, extra_history=None):
        X = self.builder.transform(future_df, extra_history=extra_history)
        X = X.reindex(columns=self.feature_columns_, fill_value=0)
        X_np = X.values.astype(float)
        X_scaled = (X_np - self._feature_mean) / self._feature_std
        X_design = np.concatenate([np.ones((X_scaled.shape[0], 1)), X_scaled], axis=1)

        eta_samples = X_design @ self.coef_samples_.T  # (n_obs, n_samples)
        mu_samples = np.exp(np.clip(eta_samples, -20, 20))

        rng = np.random.default_rng(self.seed)
        # Distribution predictive complete : pour chaque tirage posterieur
        # de mu, on tire un comptage de Poisson -> capture a la fois
        # l'incertitude des parametres ET la variabilite intrinseque du
        # comptage.
        pred_samples = rng.poisson(mu_samples)

        mean = mu_samples.mean(axis=1)
        lower = np.percentile(pred_samples, 2.5, axis=1)
        upper = np.percentile(pred_samples, 97.5, axis=1)
        return ForecastResult(mean, lower, upper)


# ---------------------------------------------------------------------------
# Registre des modeles (point d'extension)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    LinearRegressionModel.name: LinearRegressionModel,
    SarimaModel.name: SarimaModel,
    ProphetModel.name: ProphetModel,
    XGBoostModel.name: XGBoostModel,
    PoissonRegressionModel.name: PoissonRegressionModel,
    BayesianPoissonModel.name: BayesianPoissonModel,
}


# ---------------------------------------------------------------------------
# Metriques de performance
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    """Calcule RMSE, MAE, MAPE (%) et R2 entre valeurs reelles et predites."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan

    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan

    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2}


def compute_bayesian_coverage(y_true, lower, upper) -> dict:
    """
    Pour un modele fournissant un intervalle (Bayesien ou GLM Poisson) :
        - couverture : % des vraies valeurs effectivement dans [lower, upper]
          (cible ~95% si l'intervalle est a 95%)
        - largeur moyenne de l'intervalle (plus c'est etroit, mieux c'est,
          a couverture egale)
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return {
        "Couverture IC95% (%)": float(np.mean(inside) * 100),
        "Largeur moyenne IC": float(np.mean(upper - lower)),
    }


def evaluate_model(model: BaseForecastModel, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Entraine un modele sur train_df, predit sur test_df et retourne un
    dictionnaire pret a etre affiche : metriques + (si applicable)
    couverture de l'intervalle + les predictions elles-memes (pour trace).
    """
    model.fit(train_df)
    result = model.predict(test_df, extra_history=train_df[["date", "heure_debut", "volume"]])
    y_true = test_df["volume"].values
    metrics = compute_metrics(y_true, result.pred_mean)

    has_interval = result.pred_lower is not None and not np.all(np.isnan(result.pred_lower))
    if has_interval:
        metrics.update(compute_bayesian_coverage(y_true, result.pred_lower, result.pred_upper))

    return {
        "model": model,
        "metrics": metrics,
        "predictions": result.as_dataframe(),
        "has_interval": has_interval,
    }


# ---------------------------------------------------------------------------
# Recommandation de staffing
# ---------------------------------------------------------------------------

@dataclass
class StaffingRecommendation:
    agents_base: int
    agents_recommandes: int
    attente_estimee_sec: float
    taux_decroche_estime: float
    depasse_max_agents: bool


def recommend_staffing(
    volume_pred: float,
    dmt_moyenne_sec: float,
    attente_historique_sec: float,
    decroche_historique: float,
    slot_duration_sec: float = 3600.0,
    seuil_attente_sec: float = 20.0,
    seuil_decroche: float = 0.05,
    max_agents: int = 20,
) -> StaffingRecommendation:
    """
    Calcule le nombre d'agents recommande pour une tranche horaire donnee.

    Etape 1 (charge de travail) :
        agents_base = CEIL( (volume_pred * dmt_moyenne_sec) / slot_duration_sec )
        -> nombre d'agents necessaires pour ecouler la charge de travail
           totale (en "secondes-agent") sur la duree de la tranche.

    Etape 2 (ajustement qualite de service) :
        On suppose que l'attente moyenne et le taux de decroche observes
        historiquement evoluent (approximativement) en proportion inverse
        du nombre d'agents par rapport au besoin theorique de base :
            attente_estimee(N)  = attente_historique * (agents_base / N)
            decroche_estime(N)  = decroche_historique * (agents_base / N)
        C'est une approximation simple (lineaire) de l'effet du
        sureffectif/sous-effectif sur la qualite de service -- une vraie
        modelisation de file d'attente (loi d'Erlang C) serait plus precise
        mais sort du cadre de ce prototype.
        On augmente N tant que l'un des deux seuils n'est pas respecte,
        jusqu'a max_agents.

    Returns
    -------
    StaffingRecommendation
    """
    agents_base = max(1, math.ceil((volume_pred * dmt_moyenne_sec) / slot_duration_sec))

    n = agents_base
    while n <= max_agents:
        ratio = agents_base / n
        attente_est = attente_historique_sec * ratio
        decroche_est = decroche_historique * ratio
        if attente_est <= seuil_attente_sec and decroche_est <= seuil_decroche:
            break
        n += 1

    depasse = n > max_agents
    n_final = min(n, max_agents)
    ratio_final = agents_base / n_final if n_final else 1.0

    return StaffingRecommendation(
        agents_base=agents_base,
        agents_recommandes=n_final,
        attente_estimee_sec=round(attente_historique_sec * ratio_final, 1),
        taux_decroche_estime=round(min(decroche_historique * ratio_final, 1.0), 4),
        depasse_max_agents=depasse,
    )


def estimate_future_dmt_and_quality(
    agg: pd.DataFrame, future_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Pour chaque ligne de future_df (date, heure_debut, jour_semaine),
    estime la DMT moyenne, l'attente moyenne et le taux de decroche
    "attendus" en se basant sur la moyenne historique du meme couple
    (jour_semaine, heure_debut). A defaut d'historique pour ce couple
    precis, on retombe sur la moyenne globale du slot, puis sur la
    moyenne globale generale.
    """
    by_slot_day = agg.groupby(["jour_semaine", "heure_debut"]).agg(
        dmt_moyenne=("dmt_moyenne", "mean"),
        attente_moyenne=("attente_moyenne", "mean"),
        taux_decroche=("taux_decroche", "mean"),
    )
    by_slot = agg.groupby("heure_debut").agg(
        dmt_moyenne=("dmt_moyenne", "mean"),
        attente_moyenne=("attente_moyenne", "mean"),
        taux_decroche=("taux_decroche", "mean"),
    )
    global_means = agg[["dmt_moyenne", "attente_moyenne", "taux_decroche"]].mean()

    rows = []
    for _, row in future_df.iterrows():
        key = (row["jour_semaine"], row["heure_debut"])
        if key in by_slot_day.index:
            vals = by_slot_day.loc[key]
        elif row["heure_debut"] in by_slot.index:
            vals = by_slot.loc[row["heure_debut"]]
        else:
            vals = global_means
        rows.append({
            "dmt_moyenne": float(vals["dmt_moyenne"]),
            "attente_moyenne": float(vals["attente_moyenne"]),
            "taux_decroche": float(vals["taux_decroche"]),
        })
    return pd.DataFrame(rows)
