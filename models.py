"""
models.py
=========
Modèles de prévision du volume d'appels par date et tranche horaire.

Pour ajouter un modèle : créer une classe héritant de `BaseForecastModel`,
implémenter `_fit` et `_predict`, puis l'ajouter à `MODEL_REGISTRY`. L'application
lit ce registre et n'a pas besoin d'être modifiée.

Un volume d'appels par tranche contient une part de hasard qu'aucun modèle ne peut
prévoir, même en connaissant le vrai rythme d'appels. `noise_limits` estime ce
plafond : c'est la référence à laquelle comparer les scores.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Features et intervalles partagés
# ---------------------------------------------------------------------------

class TabularFeatureBuilder:
    """
    Features calendaires : indicatrices du jour et de la tranche, plus une tendance
    optionnelle. La première catégorie de chaque variable sert de référence : garder
    toutes les indicatrices avec une constante rendrait le modèle non identifiable.

    La tendance est désactivée par défaut : sur quelques mois d'historique, la
    prolonger dans le futur dégrade la prévision sans contrepartie mesurable.
    """

    def __init__(self, use_trend: bool = False):
        self.use_trend = use_trend
        self.weekdays_: list = []
        self.slots_: list = []
        self.date_ref_ = None

    def fit(self, train_df: pd.DataFrame) -> "TabularFeatureBuilder":
        self.weekdays_ = sorted(train_df["jour_semaine_num"].unique())
        self.slots_ = sorted(train_df["heure_debut"].unique())
        self.date_ref_ = pd.to_datetime(train_df["date"]).min()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = {}
        if self.use_trend:
            features["trend"] = (pd.to_datetime(df["date"]) - self.date_ref_).dt.days.astype(float)
        for weekday in self.weekdays_[1:]:
            features[f"jour_{weekday}"] = (df["jour_semaine_num"] == weekday).astype(float)
        for slot in self.slots_[1:]:
            features[f"slot_{slot}"] = (df["heure_debut"] == slot).astype(float)
        return pd.DataFrame(features, index=df.index)


def dispersion_factor(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    """Facteur phi tel que variance = phi × moyenne, au moins 1 (cas Poisson)."""
    mean = np.clip(y_fit, 1e-6, None)
    return max(float(np.sum((y_true - mean) ** 2) / np.sum(mean)), 1.0)


def count_interval(mean: np.ndarray, phi: float = 1.0, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Intervalle de prédiction d'un comptage : il doit contenir le volume réellement
    observé, pas seulement sa moyenne. Loi de Poisson si phi = 1, sinon binomiale
    négative de variance phi × moyenne.
    """
    mean = np.clip(np.asarray(mean, dtype=float), 1e-6, None)
    alpha = (1 - level) / 2
    if phi <= 1.0 + 1e-6:
        return stats.poisson.ppf(alpha, mean), stats.poisson.ppf(1 - alpha, mean)
    size, prob = mean / (phi - 1), 1 / phi
    return stats.nbinom.ppf(alpha, size, prob), stats.nbinom.ppf(1 - alpha, size, prob)


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Sortie de tous les modèles : moyenne et intervalle de prédiction à 95 %."""
    pred_mean: np.ndarray
    pred_lower: np.ndarray
    pred_upper: np.ndarray

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "pred_mean": self.pred_mean,
            "pred_lower": self.pred_lower,
            "pred_upper": self.pred_upper,
        })


class BaseForecastModel:
    """`train_df` et `future_df` ont les colonnes de `utils.aggregate_by_slot`."""
    name: str = "BaseModel"
    description: str = ""

    def fit(self, train_df: pd.DataFrame) -> "BaseForecastModel":
        self._fit(train_df)
        self._fitted = True
        return self

    def predict(self, future_df: pd.DataFrame) -> ForecastResult:
        if not getattr(self, "_fitted", False):
            raise RuntimeError(f"Le modèle {self.name} doit être entraîné avant predict().")
        return self._predict(future_df)

    def _fit(self, train_df: pd.DataFrame):
        raise NotImplementedError

    def _predict(self, future_df: pd.DataFrame) -> ForecastResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Référence
# ---------------------------------------------------------------------------

class SeasonalAverageModel(BaseForecastModel):
    name = "Référence (moyenne 4 semaines)"
    description = (
        "Moyenne du même jour et de la même tranche sur les 4 dernières semaines. "
        "Sans paramètre à entraîner, c'est le score à battre : un modèle plus "
        "complexe ne se justifie que s'il fait mieux."
    )

    def __init__(self, weeks: int = 4):
        self.weeks = weeks

    def _fit(self, train_df):
        recent = train_df[train_df["date"] > train_df["date"].max() - pd.Timedelta(weeks=self.weeks)]
        self.means_ = recent.groupby(["jour_semaine_num", "heure_debut"])["volume"].mean().rename("mean")
        self.slot_means_ = train_df.groupby("heure_debut")["volume"].mean().rename("slot_mean")
        self.global_mean_ = float(train_df["volume"].mean())

    def _predict(self, future_df):
        keys = future_df[["jour_semaine_num", "heure_debut"]]
        out = keys.merge(self.means_.reset_index(), on=["jour_semaine_num", "heure_debut"], how="left")
        out = out.merge(self.slot_means_.reset_index(), on="heure_debut", how="left")
        mean = out["mean"].fillna(out["slot_mean"]).fillna(self.global_mean_).to_numpy()
        return ForecastResult(mean, *count_interval(mean))


# ---------------------------------------------------------------------------
# Régression linéaire
# ---------------------------------------------------------------------------

class LinearRegressionModel(BaseForecastModel):
    name = "Régression linéaire"
    description = (
        "Rapide et interprétable, mais additive : elle ajoute l'effet du jour à celui "
        "de l'heure, alors que ces effets se multiplient. Elle surestime donc les "
        "heures creuses du week-end."
    )

    def __init__(self):
        self.builder = TabularFeatureBuilder()
        self.model = LinearRegression()

    def _fit(self, train_df):
        self.builder.fit(train_df)
        X = self.builder.transform(train_df)
        y = train_df["volume"].to_numpy(dtype=float)
        self.model.fit(X, y)
        self.phi_ = dispersion_factor(y, np.clip(self.model.predict(X), 0, None))

    def _predict(self, future_df):
        mean = np.clip(self.model.predict(self.builder.transform(future_df)), 0, None)
        return ForecastResult(mean, *count_interval(mean, self.phi_))


# ---------------------------------------------------------------------------
# Séries temporelles : un modèle par tranche
# ---------------------------------------------------------------------------

class SarimaModel(BaseForecastModel):
    name = "SARIMA"
    description = (
        "Un modèle par tranche horaire, avec une saisonnalité hebdomadaire. Il ne "
        "partage aucune information entre les tranches et dispose de peu "
        "d'observations par tranche sur quelques mois d'historique."
    )

    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.slot_models_ = {}

    def _fit(self, train_df):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.global_mean_ = float(train_df["volume"].mean())
        self.last_date_ = pd.to_datetime(train_df["date"]).max()
        for slot, group in train_df.groupby("heure_debut"):
            series = group.set_index(pd.to_datetime(group["date"]))["volume"].astype(float)
            series = series.asfreq("D").interpolate(limit_direction="both")  # jours fériés retirés
            if len(series) < 21 or series.nunique() <= 1:
                continue
            try:
                self.slot_models_[slot] = SARIMAX(
                    series, order=self.order, seasonal_order=self.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            except Exception:
                continue

    def _predict(self, future_df):
        n_rows = len(future_df)
        mean = np.full(n_rows, self.global_mean_)
        lower, upper = count_interval(mean)
        dates = pd.to_datetime(future_df["date"]).to_numpy()
        slots = future_df["heure_debut"].to_numpy()
        steps = ((dates - np.datetime64(self.last_date_)) / np.timedelta64(1, "D")).astype(int)

        for slot, model in self.slot_models_.items():
            rows = np.where(slots == slot)[0]
            if rows.size == 0:
                continue
            forecast = model.get_forecast(steps=int(steps[rows].max()))
            index = steps[rows] - 1
            interval = forecast.conf_int(alpha=0.05).to_numpy()
            mean[rows] = forecast.predicted_mean.to_numpy()[index]
            lower[rows] = interval[index, 0]
            upper[rows] = interval[index, 1]
        return ForecastResult(np.clip(mean, 0, None), np.clip(lower, 0, None), np.clip(upper, 0, None))


class ProphetModel(BaseForecastModel):
    name = "Prophet"
    description = (
        "Un modèle Prophet par tranche horaire, sans tendance : sur un historique "
        "court, sa tendance par défaut sur-ajuste les derniers jours."
    )

    def __init__(self):
        self.slot_models_ = {}

    def _fit(self, train_df):
        from prophet import Prophet

        self.global_mean_ = float(train_df["volume"].mean())
        for slot, group in train_df.groupby("heure_debut"):
            data = pd.DataFrame({"ds": pd.to_datetime(group["date"]), "y": group["volume"].astype(float)})
            if len(data) < 14 or data["y"].nunique() <= 1:
                continue
            model = Prophet(
                growth="flat", weekly_seasonality=True, yearly_seasonality=False,
                daily_seasonality=False, interval_width=0.95,
            )
            try:
                self.slot_models_[slot] = model.fit(data)
            except Exception:
                continue

    def _predict(self, future_df):
        mean = np.full(len(future_df), self.global_mean_)
        lower, upper = count_interval(mean)
        dates = pd.to_datetime(future_df["date"])
        slots = future_df["heure_debut"].to_numpy()

        for slot, model in self.slot_models_.items():
            rows = np.where(slots == slot)[0]
            if rows.size == 0:
                continue
            ds = dates.iloc[rows].reset_index(drop=True)
            forecast = model.predict(pd.DataFrame({"ds": ds})).set_index("ds").reindex(ds)
            mean[rows] = forecast["yhat"].to_numpy()
            lower[rows] = forecast["yhat_lower"].to_numpy()
            upper[rows] = forecast["yhat_upper"].to_numpy()
        return ForecastResult(np.clip(mean, 0, None), np.clip(lower, 0, None), np.clip(upper, 0, None))


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------

class XGBoostModel(BaseForecastModel):
    name = "XGBoost (Poisson)"
    description = (
        "Gradient boosting pour données de comptage. Utile quand les effets calendaires "
        "interagissent de façon complexe, mais moins précis qu'un GLM quand le jour et "
        "l'heure ont des effets simplement multiplicatifs."
    )

    def __init__(self, n_estimators: int = 150, max_depth: int = 3, learning_rate: float = 0.08):
        import xgboost as xgb

        self.builder = TabularFeatureBuilder()
        self.model = xgb.XGBRegressor(
            objective="count:poisson", n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.9, colsample_bytree=0.9, random_state=42,
            n_jobs=1,  # le jeu est petit : plusieurs threads ralentissent nettement l'entraînement
        )

    def _fit(self, train_df):
        self.builder.fit(train_df)
        self.model.fit(self.builder.transform(train_df), train_df["volume"].to_numpy(dtype=float))

    def _predict(self, future_df):
        mean = np.clip(self.model.predict(self.builder.transform(future_df)), 0, None)
        return ForecastResult(mean, *count_interval(mean))


# ---------------------------------------------------------------------------
# Modèles probabilistes
# ---------------------------------------------------------------------------

class PoissonRegressionModel(BaseForecastModel):
    name = "Régression de Poisson (GLM)"
    description = (
        "Adaptée aux comptages et multiplicative : effet du jour × effet de l'heure. "
        "Son intervalle de prédiction inclut l'aléa du comptage, et s'élargit "
        "automatiquement en cas de sur-dispersion."
    )

    def __init__(self):
        self.builder = TabularFeatureBuilder()

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        import statsmodels.api as sm

        return sm.add_constant(self.builder.transform(df), has_constant="add").astype(float)

    def _fit(self, train_df):
        import statsmodels.api as sm

        self.builder.fit(train_df)
        X = self._design(train_df)
        y = train_df["volume"].to_numpy(dtype=float)
        self.result_ = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        self.phi_ = dispersion_factor(y, np.asarray(self.result_.predict(X)))

    def _predict(self, future_df):
        mean = np.asarray(self.result_.predict(self._design(future_df)))
        return ForecastResult(mean, *count_interval(mean, self.phi_))


class BayesianPoissonModel(BaseForecastModel):
    name = "Poisson bayésien (PyMC)"
    description = (
        "Même structure que le GLM, avec une incertitude complète sur les paramètres. "
        "Beaucoup plus lent : il n'apporte un gain que sur un historique très court."
    )

    def __init__(self, draws: int = 1000, tune: int = 1000, chains: int = 2, seed: int = 42):
        self.builder = TabularFeatureBuilder()
        self.draws, self.tune, self.chains, self.seed = draws, tune, chains, seed
        self.diagnostics: dict = {}

    def _fit(self, train_df):
        import arviz as az
        import pymc as pm

        self.builder.fit(train_df)
        X = self.builder.transform(train_df).to_numpy(dtype=float)
        y = train_df["volume"].to_numpy(dtype=int)

        with pm.Model():
            intercept = pm.Normal("intercept", mu=np.log(max(y.mean(), 1)), sigma=2.0)
            betas = pm.Normal("betas", mu=0.0, sigma=1.5, shape=X.shape[1])
            pm.Poisson("y_obs", mu=pm.math.exp(intercept + pm.math.dot(X, betas)), observed=y)
            trace = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains, cores=1,
                random_seed=self.seed, progressbar=False, target_accept=0.9,
            )

        posterior = trace.posterior
        self.intercepts_ = posterior["intercept"].values.reshape(-1)
        self.betas_ = posterior["betas"].values.reshape(-1, X.shape[1])
        rhat = az.rhat(trace)
        self.diagnostics = {
            "rhat_max": float(max(float(rhat[name].max()) for name in rhat.data_vars)),
            "divergences": int(trace.sample_stats["diverging"].sum()),
        }

    def _predict(self, future_df):
        X = self.builder.transform(future_df).to_numpy(dtype=float)
        eta = self.intercepts_[None, :] + X @ self.betas_.T
        mu = np.exp(np.clip(eta, -20, 20))
        draws = np.random.default_rng(self.seed).poisson(mu)
        return ForecastResult(
            mu.mean(axis=1), np.percentile(draws, 2.5, axis=1), np.percentile(draws, 97.5, axis=1)
        )


MODEL_REGISTRY = {
    model.name: model
    for model in (
        SeasonalAverageModel, LinearRegressionModel, PoissonRegressionModel,
        XGBoostModel, SarimaModel, ProphetModel, BayesianPoissonModel,
    )
}
DEFAULT_MODEL = PoissonRegressionModel.name
SLOW_MODELS = {BayesianPoissonModel.name}


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, dates=None) -> dict:
    """
    RMSE, MAE, WAPE et R². Le WAPE (erreur totale / volume total) remplace le MAPE,
    qui exclut les tranches sans appel et explose sur les tranches à faible volume.
    Avec `dates`, ajoute le WAPE sur les totaux par jour.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    total = y_true.sum()
    metrics = {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
        "WAPE (%)": float(np.abs(errors).sum() / total * 100) if total else np.nan,
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    if dates is not None and total:
        daily = pd.DataFrame({"date": dates, "true": y_true, "pred": y_pred}).groupby("date").sum()
        metrics["WAPE jour (%)"] = float((daily["true"] - daily["pred"]).abs().sum() / total * 100)
    return metrics


def compute_interval_metrics(y_true, lower, upper) -> dict:
    """Part des valeurs réelles dans l'intervalle (cible : 95 %) et largeur moyenne."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return {
        "Couverture IC95% (%)": float(np.mean((y_true >= lower) & (y_true <= upper)) * 100),
        "Largeur moyenne IC": float(np.mean(upper - lower)),
    }


def noise_limits(y_true, expected) -> dict:
    """
    Meilleur score atteignable si `expected` était le vrai rythme d'appels : un
    comptage de Poisson s'en écarte par simple hasard, de façon imprévisible.
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(expected, dtype=float), 1e-9, None)
    mode = np.floor(mu)
    mean_abs_dev = 2 * (mode + 1) * stats.poisson.pmf(mode + 1, mu)  # E|X - mu| exact
    variance = y_true.var()
    return {
        "RMSE": float(np.sqrt(mu.mean())),
        "MAE": float(mean_abs_dev.mean()),
        "WAPE (%)": float(mean_abs_dev.sum() / y_true.sum() * 100) if y_true.sum() else np.nan,
        "R2": float(1 - mu.mean() / variance) if variance > 0 else np.nan,
    }


def walk_forward_evaluate(model_cls, agg: pd.DataFrame, test_days: int = 14, n_folds: int = 3) -> dict:
    """
    Validation glissante : le modèle est réentraîné à plusieurs dates de coupure et
    testé sur les `test_days` jours suivants. Il n'utilise jamais de données futures.
    Le score ne dépend pas d'une seule quinzaine de test, parfois favorable.
    """
    last_date = agg["date"].max()
    tests, predictions = [], []
    for fold in reversed(range(n_folds)):
        end = last_date - pd.Timedelta(days=test_days * fold)
        cutoff = end - pd.Timedelta(days=test_days)
        train = agg[agg["date"] <= cutoff]
        test = agg[(agg["date"] > cutoff) & (agg["date"] <= end)]
        if train["date"].nunique() < 21 or test.empty:
            continue
        model = model_cls().fit(train)
        predictions.append(model.predict(test).as_dataframe())
        tests.append(test)

    if not tests:
        raise ValueError("Historique trop court : réduisez le nombre de jours de test ou de replis.")

    test_df = pd.concat(tests, ignore_index=True)
    predictions_df = pd.concat(predictions, ignore_index=True)
    y_true = test_df["volume"].to_numpy()
    metrics = compute_metrics(y_true, predictions_df["pred_mean"], test_df["date"])
    metrics.update(compute_interval_metrics(y_true, predictions_df["pred_lower"], predictions_df["pred_upper"]))
    return {
        "metrics": metrics,
        "predictions": predictions_df,
        "test_df": test_df,
        "n_folds": len(tests),
        "diagnostics": getattr(model, "diagnostics", {}),
    }
