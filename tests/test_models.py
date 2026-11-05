import numpy as np
import pandas as pd
import pytest

from data_generator import generate_synthetic_calls
from models import (
    MODEL_REGISTRY,
    PoissonRegressionModel,
    SeasonalAverageModel,
    TabularFeatureBuilder,
    compute_metrics,
    count_interval,
    dispersion_factor,
    noise_limits,
    walk_forward_evaluate,
)
from utils import add_time_features, aggregate_by_slot


@pytest.fixture(scope="module")
def agg():
    raw = generate_synthetic_calls("2026-01-01", "2026-03-31", random_seed=7)
    return aggregate_by_slot(add_time_features(raw))


def test_features_have_a_reference_category(agg):
    builder = TabularFeatureBuilder().fit(agg)
    n_expected = agg["jour_semaine_num"].nunique() - 1 + agg["heure_debut"].nunique() - 1
    assert builder.transform(agg).shape[1] == n_expected


def test_features_keep_the_same_columns_on_a_partial_horizon(agg):
    builder = TabularFeatureBuilder().fit(agg)
    assert list(builder.transform(agg.head(3)).columns) == list(builder.transform(agg).columns)


def test_glm_recovers_the_weekly_pattern(agg):
    train, future = agg.iloc[:-77], agg.iloc[-77:]
    prediction = PoissonRegressionModel().fit(train).predict(future)
    assert np.corrcoef(prediction.pred_mean, future["volume"])[0, 1] > 0.8


def test_glm_interval_covers_observed_counts(agg):
    result = walk_forward_evaluate(PoissonRegressionModel, agg)
    assert 90 <= result["metrics"]["Couverture IC95% (%)"] <= 99


def test_glm_beats_the_reference_on_rmse(agg):
    glm = walk_forward_evaluate(PoissonRegressionModel, agg)["metrics"]["RMSE"]
    reference = walk_forward_evaluate(SeasonalAverageModel, agg)["metrics"]["RMSE"]
    assert glm < reference


def test_walk_forward_never_trains_on_test_dates(agg):
    seen = {}

    class Spy(SeasonalAverageModel):
        def _fit(self, train_df):
            seen["last_train"] = train_df["date"].max()
            super()._fit(train_df)

        def _predict(self, future_df):
            assert future_df["date"].min() > seen["last_train"]
            return super()._predict(future_df)

    assert walk_forward_evaluate(Spy, agg, test_days=14, n_folds=3)["n_folds"] == 3


def test_walk_forward_rejects_short_history(agg):
    short = agg[agg["date"] < agg["date"].min() + pd.Timedelta(days=10)]
    with pytest.raises(ValueError):
        walk_forward_evaluate(SeasonalAverageModel, short)


def test_count_interval_widens_with_overdispersion():
    low_p, high_p = count_interval(np.array([30.0]), phi=1.0)
    low_nb, high_nb = count_interval(np.array([30.0]), phi=3.0)
    assert (high_nb - low_nb) > (high_p - low_p)


def test_dispersion_factor_is_not_driven_by_near_zero_means():
    assert dispersion_factor(np.array([1.0, 30.0]), np.array([1e-6, 30.0])) < 2


def test_noise_limit_matches_simulated_poisson():
    mu = np.repeat([0.7, 4.0, 20.0], 40000)
    y = np.random.default_rng(0).poisson(mu)
    limits = noise_limits(y, mu)
    metrics = compute_metrics(y, mu)
    assert limits["RMSE"] == pytest.approx(metrics["RMSE"], rel=0.02)
    assert limits["MAE"] == pytest.approx(metrics["MAE"], rel=0.02)


def test_wape_handles_empty_slots():
    assert compute_metrics([0, 10], [1, 9])["WAPE (%)"] == pytest.approx(20.0)


def test_day_wape_lets_slot_errors_offset():
    metrics = compute_metrics([5, 5], [6, 4], dates=["d", "d"])
    assert metrics["WAPE jour (%)"] == 0.0


@pytest.mark.parametrize("name", [n for n in MODEL_REGISTRY if "bayésien" not in n])
def test_every_fast_model_returns_ordered_intervals(agg, name):
    train, future = agg.iloc[:-22], agg.iloc[-22:]
    result = MODEL_REGISTRY[name]().fit(train).predict(future)
    assert len(result.pred_mean) == len(future)
    assert np.all(result.pred_lower <= result.pred_upper)
    assert np.all(result.pred_mean >= 0)
