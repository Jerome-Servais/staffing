"""
data_generator.py
==================
Generation de donnees synthetiques realistes pour demontrer l'application
sans avoir besoin de donnees reelles (page "Donnees synthetiques").

Logique :
    - Le volume d'appels par tranche horaire suit un processus de Poisson
      dont le lambda varie par jour de la semaine et par heure (profil
      d'activite plus fort en matinee/debut d'apres-midi, plus faible le
      week-end).
    - Les horodatages individuels des appels sont tires uniformement dans
      chaque tranche.
    - Le temps d'attente suit une loi exponentielle (modele classique de
      files d'attente), tronquee pour eviter des valeurs extremes.
    - La DMT (post-appel + communication) varie par agent, avec une
      tendance a l'amelioration progressive dans le temps (apprentissage).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from utils import STATUT_NON_REPONDU, STATUT_REPONDU


DEFAULT_HOURLY_PROFILE = {
    8: 0.5, 9: 0.9, 10: 1.0, 11: 1.0, 12: 0.6, 13: 0.5,
    14: 0.9, 15: 1.0, 16: 0.95, 17: 0.7, 18: 0.4,
}

DEFAULT_WEEKDAY_PROFILE = {
    0: 1.1,   # lundi : pic d'appels frequent
    1: 1.0,
    2: 0.95,
    3: 0.95,
    4: 0.9,
    5: 0.3,   # samedi
    6: 0.15,  # dimanche
}


def generate_synthetic_calls(
    start_date: str,
    end_date: str,
    agents: Sequence[str] = ("A001", "A002", "A003", "A004", "A005"),
    base_lambda: float = 18.0,
    hourly_profile: Optional[dict] = None,
    weekday_profile: Optional[dict] = None,
    mean_wait_sec: float = 25.0,
    wait_threshold_for_abandon_sec: float = 60.0,
    mean_talk_sec: float = 210.0,
    mean_post_call_sec: float = 45.0,
    agent_skill_spread: float = 0.25,
    learning_curve: bool = True,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Genere un DataFrame de niveau "appel individuel" pret a etre exporte
    en CSV avec les colonnes attendues par l'application.

    Parameters
    ----------
    start_date, end_date : str ("YYYY-MM-DD"), bornes (incluses) de la
        periode simulee.
    agents : liste des identifiants d'agents disponibles.
    base_lambda : nombre moyen d'appels par heure pour une tranche/jour
        "de reference" (profil = 1.0), avant application des profils.
    hourly_profile : dict {heure: multiplicateur}. Heures hors plage =
        centre d'appels ferme (aucun appel genere).
    weekday_profile : dict {0=lundi..6=dimanche: multiplicateur}.
    mean_wait_sec : temps d'attente moyen cible (loi exponentielle).
    wait_threshold_for_abandon_sec : au-dela de ce temps d'attente simule,
        l'appel est considere comme abandonne (statut = non_repondu).
    mean_talk_sec, mean_post_call_sec : moyennes (loi log-normale) pour
        le temps de communication et le temps de post-appel.
    agent_skill_spread : ecart-type relatif du "niveau" de chaque agent
        (un agent plus rapide aura une DMT plus faible en moyenne).
    learning_curve : si True, la DMT diminue legerement au fil du temps
        (apprentissage progressif des agents).
    random_seed : graine aleatoire pour la reproductibilite.

    Returns
    -------
    pd.DataFrame avec les colonnes :
        id_appel, date_heure, statut, post_appel, temps_communication,
        temps_attente, id_agent
    """
    rng = np.random.default_rng(random_seed)
    hourly_profile = hourly_profile or DEFAULT_HOURLY_PROFILE
    weekday_profile = weekday_profile or DEFAULT_WEEKDAY_PROFILE

    dates = pd.date_range(start_date, end_date, freq="D")
    n_days_total = max(len(dates), 1)

    # Niveau de competence individuel de chaque agent (multiplicateur de DMT)
    agent_skill = {
        a: float(np.clip(rng.normal(1.0, agent_skill_spread), 0.6, 1.6))
        for a in agents
    }

    rows = []
    call_id = 1

    for day_idx, day in enumerate(dates):
        weekday_mult = weekday_profile.get(day.weekday(), 1.0)
        # Progression de l'apprentissage : -20% de DMT sur toute la periode simulee
        learning_mult = (
            1.0 - 0.20 * (day_idx / max(n_days_total - 1, 1)) if learning_curve else 1.0
        )

        for hour, hour_mult in hourly_profile.items():
            lam = max(base_lambda * hour_mult * weekday_mult, 0.05)
            n_calls = rng.poisson(lam)
            if n_calls == 0:
                continue

            slot_start = pd.Timestamp(day) + pd.Timedelta(hours=hour)
            offsets_seconds = rng.uniform(0, 3600, size=n_calls)
            timestamps = slot_start + pd.to_timedelta(offsets_seconds, unit="s")

            waits = rng.exponential(mean_wait_sec, size=n_calls)
            assigned_agents = rng.choice(agents, size=n_calls)

            for i in range(n_calls):
                wait = float(waits[i])
                agent = assigned_agents[i]
                abandoned = wait > wait_threshold_for_abandon_sec

                if abandoned:
                    rows.append({
                        "id_appel": f"C{call_id:06d}",
                        "date_heure": timestamps[i],
                        "statut": STATUT_NON_REPONDU,
                        "post_appel": np.nan,
                        "temps_communication": np.nan,
                        "temps_attente": round(wait, 1),
                        "id_agent": "",
                    })
                else:
                    skill = agent_skill[agent] * learning_mult
                    talk = float(rng.lognormal(
                        mean=np.log(mean_talk_sec * skill), sigma=0.35
                    ))
                    post_call = float(rng.lognormal(
                        mean=np.log(mean_post_call_sec * skill), sigma=0.35
                    ))
                    rows.append({
                        "id_appel": f"C{call_id:06d}",
                        "date_heure": timestamps[i],
                        "statut": STATUT_REPONDU,
                        "post_appel": round(post_call, 1),
                        "temps_communication": round(talk, 1),
                        "temps_attente": round(wait, 1),
                        "id_agent": agent,
                    })
                call_id += 1

    df = pd.DataFrame(rows).sort_values("date_heure").reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Petit jeu de demo : 90 jours, 5 agents
    demo = generate_synthetic_calls("2026-01-01", "2026-03-31")
    demo.to_csv("sample_data.csv", index=False)
    print(f"{len(demo)} appels generes -> sample_data.csv")
