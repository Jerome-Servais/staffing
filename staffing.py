"""
staffing.py
===========
Dimensionnement d'un centre d'appels par la théorie des files d'attente.

Le nombre d'agents n'est pas proportionnel à la charge : en dessous d'un certain
seuil, l'attente explose, et au-dessus, chaque agent supplémentaire apporte très
peu. La loi d'Erlang C capture cette relation à partir de trois données seulement :
le volume d'appels, la durée moyenne de traitement (DMT) et le nombre d'agents.

Hypothèses d'Erlang C : arrivées de Poisson, file d'attente sans limite, aucun
abandon, durées de service exponentielles. L'absence d'abandon et des durées moins
variables qu'une exponentielle rendent le résultat plutôt prudent, ce que
`simulate_queue_kpis` permet de vérifier sur chaque recommandation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formules d'Erlang
# ---------------------------------------------------------------------------

def erlang_b(n_agents: int, charge: float) -> float:
    """Probabilité de blocage d'Erlang B (récurrence stable numériquement)."""
    blocage = 1.0
    for k in range(1, n_agents + 1):
        blocage = charge * blocage / (k + charge * blocage)
    return blocage


def erlang_c(n_agents: int, charge: float) -> float:
    """Probabilité qu'un appel doive attendre, pour `charge` erlangs."""
    if n_agents <= 0:
        return 1.0
    if charge <= 0:
        return 0.0
    if n_agents <= charge:
        return 1.0  # file instable : l'attente croît sans limite
    blocage = erlang_b(n_agents, charge)
    return n_agents * blocage / (n_agents - charge * (1 - blocage))


def service_level(n_agents: int, charge: float, dmt_sec: float, delai_sec: float) -> float:
    """Part des appels pris en moins de `delai_sec` secondes."""
    if charge <= 0:
        return 1.0
    if n_agents <= charge:
        return 0.0
    attente = erlang_c(n_agents, charge)
    return 1.0 - attente * math.exp(-(n_agents - charge) * delai_sec / dmt_sec)


def average_speed_of_answer(n_agents: int, charge: float, dmt_sec: float) -> float:
    """Attente moyenne (secondes), appels pris immédiatement compris."""
    if charge <= 0:
        return 0.0
    if n_agents <= charge:
        return math.inf
    return erlang_c(n_agents, charge) * dmt_sec / (n_agents - charge)


# ---------------------------------------------------------------------------
# Recommandation
# ---------------------------------------------------------------------------

@dataclass
class StaffingRecommendation:
    charge_erlangs: float
    agents_recommandes: int
    niveau_service_estime: float
    attente_estimee_sec: float
    occupation_estimee: float
    depasse_max_agents: bool


def recommend_staffing(
    volume_pred: float,
    dmt_sec: float,
    slot_duration_sec: float = 3600.0,
    objectif_service: float = 0.80,
    delai_reponse_sec: float = 20.0,
    occupation_max: float = 0.85,
    max_agents: int = 20,
    min_agents: int = 1,
) -> StaffingRecommendation:
    """
    Plus petit nombre d'agents qui respecte à la fois :
      - le niveau de service (ex. 80 % des appels pris en moins de 20 s) ;
      - l'occupation maximale, car un agent occupé à 100 % n'est pas tenable.

    Si `max_agents` ne suffit pas, le résultat est plafonné et
    `depasse_max_agents` vaut True : les indicateurs décrivent alors ce plafond.
    """
    if volume_pred < 0 or dmt_sec < 0 or slot_duration_sec <= 0:
        raise ValueError("volume_pred et dmt_sec doivent être >= 0, slot_duration_sec > 0.")

    charge = volume_pred * dmt_sec / slot_duration_sec
    n = max(min_agents, math.floor(charge) + 1)
    while (
        service_level(n, charge, dmt_sec, delai_reponse_sec) < objectif_service
        or charge / n > occupation_max
    ):
        n += 1

    depasse = n > max_agents
    n = min(n, max_agents)
    return StaffingRecommendation(
        charge_erlangs=round(charge, 3),
        agents_recommandes=n,
        niveau_service_estime=round(service_level(n, charge, dmt_sec, delai_reponse_sec), 4),
        attente_estimee_sec=round(average_speed_of_answer(n, charge, dmt_sec), 1),
        occupation_estimee=round(charge / n, 4) if n else 0.0,
        depasse_max_agents=depasse,
    )


def estimate_future_dmt(agg: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    """
    DMT attendue pour chaque ligne de `future_df` : moyenne historique du couple
    (jour, tranche), pondérée par le nombre d'appels répondus, avec repli sur la
    tranche puis sur la moyenne générale.
    """
    answered = agg[(agg["n_repondu"] > 0) & agg["dmt_moyenne"].notna()].copy()
    if answered.empty:
        raise ValueError("Aucun appel répondu : impossible d'estimer la DMT.")
    answered["duree_totale"] = answered["dmt_moyenne"] * answered["n_repondu"]

    def weighted(keys: list[str]) -> pd.DataFrame:
        grouped = answered.groupby(keys)[["duree_totale", "n_repondu"]].sum()
        return (grouped["duree_totale"] / grouped["n_repondu"]).rename("dmt").reset_index()

    keys_fine = ["jour_semaine_num", "heure_debut"]
    out = future_df[keys_fine].merge(weighted(keys_fine), on=keys_fine, how="left")
    fallback = out[["heure_debut"]].merge(weighted(["heure_debut"]), on="heure_debut", how="left")
    global_dmt = answered["duree_totale"].sum() / answered["n_repondu"].sum()
    return out["dmt"].fillna(fallback["dmt"]).fillna(global_dmt).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Simulation d'une file d'attente
# ---------------------------------------------------------------------------

def simulate_fcfs(
    arrivals_sec: np.ndarray,
    service_time: Callable[[int, int], float],
    n_servers: int,
    patience_sec: Optional[np.ndarray] = None,
    active_servers: Optional[Callable[[float], int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    File premier arrivé, premier servi, à `n_servers` agents.

    Paramètres
    ----------
    arrivals_sec : instants d'arrivée triés.
    service_time : (agent, appel) -> durée de traitement de cet appel.
    patience_sec : temps d'attente maximal de chaque appelant, puis il raccroche.
    active_servers : instant -> nombre d'agents en poste, pour un planning variable.

    Retourne (attente, agent, durée de service). `agent` vaut -1 pour un appel
    abandonné, dont l'attente est alors égale à la patience de l'appelant.
    """
    n_calls = len(arrivals_sec)
    free_at = np.zeros(n_servers)
    waits = np.zeros(n_calls)
    servers = np.full(n_calls, -1, dtype=int)
    services = np.full(n_calls, np.nan)

    for i, arrival in enumerate(arrivals_sec):
        n_active = n_servers if active_servers is None else active_servers(arrival)
        n_active = int(np.clip(n_active, 1, n_servers))
        agent = int(np.argmin(free_at[:n_active]))
        start = max(arrival, free_at[agent])
        wait = start - arrival
        if patience_sec is not None and wait > patience_sec[i]:
            waits[i] = patience_sec[i]
            continue
        duration = service_time(agent, i)
        waits[i], servers[i], services[i] = wait, agent, duration
        free_at[agent] = start + duration
    return waits, servers, services


def simulate_queue_kpis(
    volume_par_tranche: float,
    dmt_sec: float,
    n_agents: int,
    slot_duration_sec: float = 3600.0,
    delai_reponse_sec: float = 20.0,
    cv_service: float = 0.5,
    n_tranches: int = 500,
    random_seed: int = 0,
) -> dict:
    """Mesure par simulation, sans abandon, le service obtenu avec `n_agents` agents."""
    rng = np.random.default_rng(random_seed)
    duration = n_tranches * slot_duration_sec
    n_calls = rng.poisson(volume_par_tranche * n_tranches)
    if n_calls == 0:
        return {"niveau_service": 1.0, "attente_moyenne_sec": 0.0}

    arrivals = np.sort(rng.uniform(0, duration, n_calls))
    sigma = math.sqrt(math.log(1 + cv_service ** 2))
    durations = dmt_sec * rng.lognormal(-sigma ** 2 / 2, sigma, n_calls)
    waits, _, _ = simulate_fcfs(arrivals, lambda _agent, i: durations[i], n_agents)

    regime_etabli = waits[n_calls // 10:]
    return {
        "niveau_service": float(np.mean(regime_etabli <= delai_reponse_sec)),
        "attente_moyenne_sec": float(regime_etabli.mean()),
    }
