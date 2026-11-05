import numpy as np
import pandas as pd
import pytest

from utils import (
    add_time_features,
    aggregate_by_slot,
    build_future_skeleton,
    dispersion_test,
    dmt_by_agent,
    hour_label,
    kpi_summary,
    pooled_dispersion_test,
    poisson_validation_report,
)


def calls(rows):
    df = pd.DataFrame(rows, columns=["id_appel", "date_heure", "statut", "post_appel",
                                     "temps_communication", "temps_attente", "id_agent"])
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    return df


@pytest.fixture
def raw():
    return calls([
        ("1", "2026-01-05 08:10:00", "repondu", 50.0, 150.0, 10.0, "A"),
        ("2", "2026-01-05 08:20:00", "repondu", 50.0, 250.0, 30.0, "A"),
        ("3", "2026-01-05 08:30:00", "non_repondu", np.nan, np.nan, 90.0, ""),
        ("4", "2026-01-05 09:10:00", "repondu", 10.0, 90.0, 5.0, "B"),
        ("5", "2026-01-06 09:15:00", "repondu", 10.0, 90.0, 5.0, "B"),
    ])


def test_service_level_and_abandon_are_separate_kpis(raw):
    agg = aggregate_by_slot(add_time_features(raw), wait_threshold_sec=20)
    row = agg.iloc[0]
    assert row["volume"] == 3
    assert row["taux_abandon"] == pytest.approx(1 / 3)
    assert row["niveau_service"] == pytest.approx(1 / 3)  # un seul pris en moins de 20 s


def test_handle_time_excludes_waiting_and_abandoned_calls(raw):
    agg = aggregate_by_slot(add_time_features(raw))
    assert agg.iloc[0]["dmt_moyenne"] == pytest.approx(250.0)  # (200 + 300) / 2


def test_average_wait_covers_answered_calls_only(raw):
    agg = aggregate_by_slot(add_time_features(raw))
    assert agg.iloc[0]["attente_moyenne"] == pytest.approx(20.0)


def test_empty_slots_are_kept_with_zero_volume(raw):
    agg = aggregate_by_slot(add_time_features(raw))
    assert len(agg) == 4  # 2 jours x 2 tranches
    empty = agg[(agg["date"] == "2026-01-06") & (agg["heure_debut"] == 8)].iloc[0]
    assert empty["volume"] == 0
    assert np.isnan(empty["taux_abandon"])


def test_half_hour_slots_stay_distinct(raw):
    agg = aggregate_by_slot(add_time_features(raw, 30), slot_minutes=30)
    assert {8.0, 8.5, 9.0} == set(agg["heure_debut"])
    assert agg["tranche"].iloc[1].endswith("8h30-9h")
    assert hour_label(8.5) == "8h30"


def test_kpis_are_weighted_by_volume(raw):
    kpis = kpi_summary(aggregate_by_slot(add_time_features(raw), wait_threshold_sec=20))
    assert kpis["volume"] == 5
    assert kpis["taux_abandon"] == pytest.approx(1 / 5)
    assert kpis["niveau_service"] == pytest.approx(3 / 5)
    assert kpis["dmt_moyenne"] == pytest.approx(175.0)


def test_agent_handle_time_excludes_waiting(raw):
    out = dmt_by_agent(add_time_features(raw))
    agent_a = out[out["id_agent"] == "A"].iloc[0]
    assert agent_a["dmt_moyenne"] == pytest.approx(250.0)


def test_future_skeleton_supports_half_hours():
    future = build_future_skeleton(pd.Timestamp("2026-01-05"), 1, [8.0, 8.5], exclude_holidays=False, slot_minutes=30)
    assert future["tranche"].tolist() == ["2026-01-06_8h-8h30", "2026-01-06_8h30-9h"]


def panel(rng, sample):
    rows = [
        {"jour_semaine_num": day, "jour_semaine": str(day), "heure_debut": hour, "volume": value}
        for day in range(5) for hour in range(11) for value in sample(rng, 20 + 2 * hour)
    ]
    return pd.DataFrame(rows)


def test_pooled_test_keeps_its_false_positive_rate():
    rng = np.random.default_rng(0)
    rejected = [
        pooled_dispersion_test(panel(rng, lambda r, lam: r.poisson(lam, 13)))["p_value"] < 0.05
        for _ in range(200)
    ]
    assert np.mean(rejected) < 0.10


def test_pooled_test_detects_overdispersion():
    rng = np.random.default_rng(0)
    data = panel(rng, lambda r, lam: r.poisson(r.gamma(lam, 1.0, 13)))  # variance = 2 x moyenne
    assert pooled_dispersion_test(data)["p_value"] < 0.001


def test_dispersion_test_needs_enough_observations():
    assert not dispersion_test(np.array([1, 2, 3]))["valide"]


def test_report_adjusts_for_multiple_comparisons():
    report = poisson_validation_report(panel(np.random.default_rng(1), lambda r, lam: r.poisson(lam, 13)))
    assert (report["p_ajustee"] >= report["p_value"] - 1e-9).all()
