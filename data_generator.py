"""
data_generator.py
=================
Données synthétiques réalistes, pour utiliser l'application sans données réelles.

Les appels traversent une véritable file d'attente : l'attente résulte du volume
reçu et du nombre d'agents en poste, comme dans un centre d'appels. Sans ce lien,
aucune recommandation de staffing ne pourrait être vérifiée.

    - Volume : processus de Poisson, dont l'intensité dépend du jour et de l'heure.
    - Planning : effectif fixe par heure, calé sur un jour de semaine moyen. Le
      lundi est donc tendu et le week-end largement couvert.
    - Abandons : chaque appelant a une patience aléatoire et raccroche au-delà.
    - DMT : varie selon l'agent et baisse au fil du temps (apprentissage).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from staffing import simulate_fcfs
from utils import STATUT_NON_REPONDU, STATUT_REPONDU


DEFAULT_HOURLY_PROFILE = {
    8: 0.5, 9: 0.9, 10: 1.0, 11: 1.0, 12: 0.6, 13: 0.5,
    14: 0.9, 15: 1.0, 16: 0.95, 17: 0.7, 18: 0.4,
}

DEFAULT_WEEKDAY_PROFILE = {
    0: 1.1,   # lundi : jour le plus chargé
    1: 1.0,
    2: 0.95,
    3: 0.95,
    4: 0.9,
    5: 0.3,   # samedi
    6: 0.15,  # dimanche
}


def true_call_rate(
    weekday: int,
    hour: int,
    base_lambda: float = 18.0,
    hourly_profile: Optional[dict] = None,
    weekday_profile: Optional[dict] = None,
) -> float:
    """Vraie intensité horaire du générateur, utile pour évaluer les modèles."""
    hourly_profile = hourly_profile or DEFAULT_HOURLY_PROFILE
    weekday_profile = weekday_profile or DEFAULT_WEEKDAY_PROFILE
    if hour not in hourly_profile:
        return 0.0
    return max(base_lambda * hourly_profile[hour] * weekday_profile.get(weekday, 1.0), 0.05)


def build_schedule(
    hourly_profile: dict,
    weekday_profile: dict,
    base_lambda: float,
    mean_handle_sec: float,
    n_agents: int,
    target_occupancy: float = 0.75,
) -> dict:
    """Planning par heure, dimensionné pour un jour de semaine moyen."""
    weekday_mean = np.mean([weekday_profile.get(day, 1.0) for day in range(5)])
    schedule = {}
    for hour, multiplier in hourly_profile.items():
        charge = base_lambda * multiplier * weekday_mean * mean_handle_sec / 3600
        schedule[hour] = int(np.clip(math.ceil(charge / target_occupancy), 1, n_agents))
    return schedule


def generate_synthetic_calls(
    start_date: str,
    end_date: str,
    agents: Sequence[str] = ("A001", "A002", "A003", "A004", "A005"),
    base_lambda: float = 18.0,
    hourly_profile: Optional[dict] = None,
    weekday_profile: Optional[dict] = None,
    mean_patience_sec: float = 150.0,
    mean_talk_sec: float = 210.0,
    mean_post_call_sec: float = 45.0,
    agent_skill_spread: float = 0.15,
    learning_curve: bool = True,
    schedule: Optional[dict] = None,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Génère un DataFrame d'appels individuels, avec les colonnes du CSV attendu :
    id_appel, date_heure, statut, post_appel, temps_communication, temps_attente,
    id_agent.

    Paramètres
    ----------
    start_date, end_date : bornes incluses, au format "YYYY-MM-DD".
    agents : identifiants des agents disponibles.
    base_lambda : appels par heure pour un multiplicateur de 1.0.
    hourly_profile : {heure: multiplicateur}, le centre est fermé aux autres heures.
    weekday_profile : {0 = lundi ... 6 = dimanche: multiplicateur}.
    mean_patience_sec : patience moyenne des appelants (loi exponentielle).
    mean_talk_sec, mean_post_call_sec : durées moyennes, de loi log-normale.
    agent_skill_spread : dispersion du niveau des agents (multiplicateur de DMT).
    learning_curve : si True, la DMT baisse de 20 % sur la période simulée.
    schedule : {heure: agents} ou {(jour, heure): agents}, calculé si absent.
    random_seed : graine aléatoire, pour la reproductibilité.
    """
    rng = np.random.default_rng(random_seed)
    hourly_profile = hourly_profile or DEFAULT_HOURLY_PROFILE
    weekday_profile = weekday_profile or DEFAULT_WEEKDAY_PROFILE
    agents = list(agents)
    if schedule is None:
        schedule = build_schedule(
            hourly_profile, weekday_profile, base_lambda,
            mean_talk_sec + mean_post_call_sec, len(agents),
        )

    skills = np.clip(rng.normal(1.0, agent_skill_spread, len(agents)), 0.6, 1.6)
    dates = pd.date_range(start_date, end_date, freq="D")
    sigma = 0.35
    log_shift = -sigma ** 2 / 2  # pour que la moyenne soit bien celle demandée

    rows = []
    for day_index, day in enumerate(dates):
        arrivals = np.concatenate([
            hour * 3600 + rng.uniform(0, 3600, rng.poisson(
                true_call_rate(day.weekday(), hour, base_lambda, hourly_profile, weekday_profile)
            ))
            for hour in hourly_profile
        ])
        if arrivals.size == 0:
            continue
        arrivals = np.sort(arrivals)

        learning = 1.0
        if learning_curve:
            learning -= 0.20 * day_index / max(len(dates) - 1, 1)

        talks, posts = {}, {}

        def service_time(agent: int, call: int) -> float:
            speed = skills[agent] * learning
            talks[call] = mean_talk_sec * speed * rng.lognormal(log_shift, sigma)
            posts[call] = mean_post_call_sec * speed * rng.lognormal(log_shift, sigma)
            return talks[call] + posts[call]

        def active_agents(moment_sec: float, weekday: int = day.weekday()) -> int:
            hour = int(moment_sec // 3600)
            return schedule.get((weekday, hour), schedule.get(hour, 1))

        waits, servers, _ = simulate_fcfs(
            arrivals, service_time, len(agents),
            patience_sec=rng.exponential(mean_patience_sec, arrivals.size),
            active_servers=active_agents,
        )

        for call, arrival in enumerate(arrivals):
            answered = servers[call] >= 0
            rows.append({
                "date_heure": day + pd.Timedelta(seconds=round(float(arrival))),
                "statut": STATUT_REPONDU if answered else STATUT_NON_REPONDU,
                "post_appel": round(posts[call], 1) if answered else np.nan,
                "temps_communication": round(talks[call], 1) if answered else np.nan,
                "temps_attente": round(float(waits[call]), 1),
                "id_agent": agents[servers[call]] if answered else "",
            })

    df = pd.DataFrame(rows).sort_values("date_heure").reset_index(drop=True)
    df.insert(0, "id_appel", [f"C{i + 1:06d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    demo = generate_synthetic_calls("2026-01-01", "2026-03-31")
    demo.to_csv("exemple_donnees_appels.csv", index=False)
    print(f"{len(demo)} appels générés -> exemple_donnees_appels.csv")
