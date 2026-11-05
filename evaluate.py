"""
evaluate.py
===========
Évaluation reproductible, sur données synthétiques : `python evaluate.py`.

1. Prévision : score hors échantillon de chaque modèle, comparé au score d'un
   modèle qui connaîtrait le vrai rythme d'appels du générateur.
2. Staffing : apprend sur une période, recommande un planning, puis fait vivre à
   ce planning une nouvelle période simulée, avec abandons et variabilité des agents.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from data_generator import (
    DEFAULT_HOURLY_PROFILE,
    DEFAULT_WEEKDAY_PROFILE,
    build_schedule,
    generate_synthetic_calls,
    true_call_rate,
)
from models import (
    MODEL_REGISTRY,
    PoissonRegressionModel,
    compute_interval_metrics,
    compute_metrics,
    count_interval,
    walk_forward_evaluate,
)
from staffing import estimate_future_dmt, recommend_staffing
from utils import add_time_features, aggregate_by_slot, kpi_summary

AGENTS = tuple(f"A{i:03d}" for i in range(1, 11))
HISTORY = ("2026-01-05", "2026-03-29")
LIVE = ("2026-03-30", "2026-04-26")


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_slot(add_time_features(raw))


def oracle_metrics(test_df: pd.DataFrame) -> dict:
    """Score d'un modèle qui connaîtrait le vrai rythme : seul le hasard reste."""
    mean = np.array([
        true_call_rate(day, int(hour))
        for day, hour in zip(test_df["jour_semaine_num"], test_df["heure_debut"])
    ])
    metrics = compute_metrics(test_df["volume"], mean, test_df["date"])
    metrics.update(compute_interval_metrics(test_df["volume"], *count_interval(mean)))
    return metrics


def evaluate_forecasts(n_datasets: int, models: list[str]) -> pd.DataFrame:
    rows = []
    for seed in range(n_datasets):
        agg = aggregate(generate_synthetic_calls("2026-01-01", "2026-03-31", random_seed=100 + seed))
        for name in models:
            result = walk_forward_evaluate(MODEL_REGISTRY[name], agg)
            rows.append({"Modèle": name, **result["metrics"]})
        rows.append({"Modèle": "Limite (vrai rythme connu)", **oracle_metrics(result["test_df"])})
    return pd.DataFrame(rows).groupby("Modèle").mean().sort_values("RMSE")


def recommended_schedule(history: pd.DataFrame, objectif: float, delai: float) -> dict:
    """Planning par (jour, heure) issu de la prévision du GLM et d'Erlang C."""
    future = history[["jour_semaine_num", "heure_debut"]].drop_duplicates().reset_index(drop=True)
    volumes = PoissonRegressionModel().fit(history).predict(future).pred_mean
    dmt = estimate_future_dmt(history, future)
    return {
        (int(row.jour_semaine_num), int(row.heure_debut)): recommend_staffing(
            float(volumes[i]), float(dmt[i]), objectif_service=objectif,
            delai_reponse_sec=delai, max_agents=len(AGENTS),
        ).agents_recommandes
        for i, row in enumerate(future.itertuples())
    }


def default_weekly_hours() -> int:
    """Heures-agent hebdomadaires du planning par défaut du générateur."""
    plan = build_schedule(DEFAULT_HOURLY_PROFILE, DEFAULT_WEEKDAY_PROFILE, 18.0, 255.0, 5)
    return 7 * sum(plan.values())


def evaluate_staffing(n_runs: int, objectif: float = 0.80, delai: float = 20.0) -> pd.DataFrame:
    history_raw = generate_synthetic_calls(*HISTORY, agents=AGENTS[:5], learning_curve=False, random_seed=1)
    history = aggregate(history_raw)
    schedule = recommended_schedule(history, objectif, delai)

    rows, slots = [], []
    for seed in range(n_runs):
        for label, plan in (("Planning d'origine", None), ("Planning recommandé", schedule)):
            agents = AGENTS[:5] if plan is None else AGENTS
            live = aggregate(generate_synthetic_calls(
                *LIVE, agents=agents, learning_curve=False, schedule=plan, random_seed=200 + seed,
            ))
            kpis = kpi_summary(live)
            rows.append({
                "Planning": label,
                "Niveau de service (%)": kpis["niveau_service"] * 100,
                "Abandons (%)": kpis["taux_abandon"] * 100,
                "Attente moyenne (s)": kpis["attente_moyenne"],
                "Heures-agent par semaine": default_weekly_hours() if plan is None else sum(plan.values()),
            })
            slots.append(live.assign(Planning=label))

    summary = pd.DataFrame(rows).groupby("Planning").mean()
    # Tranches ouvrées uniquement : le week-end, quelques appels ne font pas un score fiable.
    weekdays = pd.concat(slots).query("jour_semaine_num < 5")
    by_slot = weekdays.groupby(["Planning", "jour_semaine_num", "heure_debut"])[["n_dans_delai", "volume"]].sum()
    worst = (by_slot["n_dans_delai"] / by_slot["volume"]).groupby("Planning").min() * 100
    summary["Pire tranche ouvrée (%)"] = worst
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=5, help="jeux de données pour la prévision")
    parser.add_argument("--runs", type=int, default=20, help="simulations du planning")
    parser.add_argument("--all-models", action="store_true", help="inclut le modèle bayésien, plus lent")
    args = parser.parse_args()

    np.random.seed(0)  # Prophet tire ses intervalles avec le générateur global
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("pymc").setLevel(logging.ERROR)
    models = [m for m in MODEL_REGISTRY if args.all_models or "bayésien" not in m]

    print("\n## Prévision (validation glissante, moyenne sur plusieurs jeux)\n")
    print(evaluate_forecasts(args.datasets, models).round(2).to_string())
    print("\n## Staffing (objectif : 80 % des appels pris en moins de 20 s)\n")
    print(evaluate_staffing(args.runs).round(1).to_string())


if __name__ == "__main__":
    main()
