import math

import numpy as np
import pandas as pd
import pytest

from data_generator import generate_synthetic_calls
from staffing import (
    average_speed_of_answer,
    erlang_c,
    estimate_future_dmt,
    recommend_staffing,
    service_level,
    simulate_fcfs,
    simulate_queue_kpis,
)
from utils import add_time_features, aggregate_by_slot


def test_erlang_c_matches_known_value():
    # Valeur de référence : 2 agents pour 1 erlang -> P(attente) = 1/3
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3)


def test_erlang_c_is_certain_when_queue_is_unstable():
    assert erlang_c(2, 2.0) == 1.0
    assert service_level(2, 2.5, 240, 20) == 0.0
    assert math.isinf(average_speed_of_answer(2, 2.5, 240))


def test_no_load_needs_no_waiting():
    assert erlang_c(3, 0.0) == 0.0
    assert service_level(3, 0.0, 240, 20) == 1.0


def test_recommendation_is_smallest_team_meeting_target():
    rec = recommend_staffing(18, 236, objectif_service=0.8, delai_reponse_sec=20)
    charge = 18 * 236 / 3600
    assert rec.agents_recommandes == 3
    assert rec.niveau_service_estime >= 0.8
    assert service_level(2, charge, 236, 20) < 0.8


def test_occupancy_limit_can_add_agents():
    loose = recommend_staffing(600, 240, objectif_service=0.5, occupation_max=1.0, max_agents=200)
    strict = recommend_staffing(600, 240, objectif_service=0.5, occupation_max=0.80, max_agents=200)
    assert strict.agents_recommandes > loose.agents_recommandes
    assert strict.occupation_estimee <= 0.80


def test_cap_is_reported_and_keeps_indicators_consistent():
    rec = recommend_staffing(600, 240, max_agents=10)
    assert rec.depasse_max_agents
    assert rec.agents_recommandes == 10
    assert rec.niveau_service_estime == 0.0


def test_zero_volume_keeps_one_agent_on_duty():
    rec = recommend_staffing(0, 240)
    assert rec.agents_recommandes == 1
    assert rec.niveau_service_estime == 1.0


def test_negative_volume_is_rejected():
    with pytest.raises(ValueError):
        recommend_staffing(-1, 240)


def test_slot_duration_changes_the_load():
    assert recommend_staffing(10, 240, slot_duration_sec=1800).charge_erlangs == pytest.approx(
        recommend_staffing(20, 240, slot_duration_sec=3600).charge_erlangs
    )


def test_simulation_agrees_with_erlang_c_for_exponential_like_service():
    # cv = 1 : hypothèse d'Erlang C. La simulation doit retrouver la formule.
    measured = simulate_queue_kpis(60, 240, 6, cv_service=1.0, n_tranches=3000)
    expected = service_level(6, 4.0, 240, 20)
    assert measured["niveau_service"] == pytest.approx(expected, abs=0.04)


def test_fcfs_queue_order_and_abandon():
    arrivals = np.array([0.0, 1.0, 2.0])
    waits, servers, _ = simulate_fcfs(
        arrivals, lambda _agent, _call: 100.0, 1, patience_sec=np.array([9.0, 200.0, 5.0])
    )
    assert waits.tolist() == [0.0, 99.0, 5.0]
    assert servers.tolist() == [0, 0, -1]


def test_generated_waits_depend_on_the_load():
    raw = generate_synthetic_calls("2026-01-01", "2026-01-31", random_seed=1)
    agg = aggregate_by_slot(add_time_features(raw))
    busy = agg[agg["jour_semaine_num"] < 5]
    quiet = agg[agg["jour_semaine_num"] >= 5]
    assert busy["taux_abandon"].mean() > quiet["taux_abandon"].mean()
    assert busy["niveau_service"].mean() < quiet["niveau_service"].mean()


def test_future_dmt_is_weighted_by_answered_calls_and_falls_back():
    agg = pd.DataFrame({
        "jour_semaine_num": [0, 0],
        "heure_debut": [8.0, 8.0],
        "n_repondu": [1, 3],
        "dmt_moyenne": [100.0, 200.0],
    })
    future = pd.DataFrame({"jour_semaine_num": [0, 3, 3], "heure_debut": [8.0, 8.0, 12.0]})
    assert estimate_future_dmt(agg, future) == pytest.approx([175.0, 175.0, 175.0])
