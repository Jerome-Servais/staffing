"""
utils.py
========
Fonctions de pretraitement, d'agregation et de validation statistique
pour l'application de prevision des appels et de recommandation de staffing.

Toutes les fonctions sont documentees (docstrings) et concues pour etre
appelees independamment de Streamlit, ce qui permet de les tester
unitairement (voir tests/test_utils.py).

Colonnes attendues dans le CSV d'entree (voir README.md) :
    - id_appel             : identifiant unique de l'appel
    - date_heure           : date et heure de l'appel (parsable par pandas)
    - statut               : "repondu" ou "non_repondu"
    - post_appel           : temps de post-appel, en secondes
    - temps_communication  : duree de la conversation, en secondes
    - temps_attente        : temps d'attente avant prise d'appel, en secondes
    - id_agent             : identifiant de l'agent ayant traite l'appel
                              (peut etre vide si statut == "non_repondu")
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    import holidays as holidays_lib
except ImportError:  # pragma: no cover
    holidays_lib = None


REQUIRED_COLUMNS = [
    "id_appel",
    "date_heure",
    "statut",
    "post_appel",
    "temps_communication",
    "temps_attente",
    "id_agent",
]

STATUT_REPONDU = "repondu"
STATUT_NON_REPONDU = "non_repondu"

JOURS_SEMAINE_FR = [
    "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"
]


# ---------------------------------------------------------------------------
# Chargement et validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Resultat de la validation d'un fichier CSV importe."""
    is_valid: bool
    missing_columns: list
    n_rows: int
    n_invalid_dates: int
    warnings: list


def load_calls_csv(file_or_buffer) -> pd.DataFrame:
    """
    Charge un fichier CSV d'appels et parse la colonne date_heure.

    Parameters
    ----------
    file_or_buffer : str, Path, or file-like object
        Chemin du fichier ou objet fichier (ex: depuis st.file_uploader).

    Returns
    -------
    pd.DataFrame
        Donnees brutes, avec date_heure converti en datetime.
    """
    df = pd.read_csv(file_or_buffer)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date_heure" in df.columns:
        df["date_heure"] = pd.to_datetime(df["date_heure"], errors="coerce")
    return df


def validate_calls_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Verifie que le DataFrame contient les colonnes requises et des dates
    valides. Ne leve pas d'exception : retourne un rapport structure pour
    affichage cote interface (utile pour un public non technique).
    """
    warnings = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    n_invalid_dates = 0
    if "date_heure" in df.columns:
        n_invalid_dates = int(df["date_heure"].isna().sum())
        if n_invalid_dates > 0:
            warnings.append(
                f"{n_invalid_dates} ligne(s) ont une date_heure illisible "
                "et seront ignorees."
            )

    if "statut" in df.columns:
        valeurs_inattendues = set(df["statut"].dropna().unique()) - {
            STATUT_REPONDU, STATUT_NON_REPONDU
        }
        if valeurs_inattendues:
            warnings.append(
                "Valeurs inattendues dans la colonne 'statut' : "
                f"{sorted(valeurs_inattendues)}. Seules 'repondu' et "
                "'non_repondu' sont reconnues."
            )

    return ValidationResult(
        is_valid=(len(missing) == 0),
        missing_columns=missing,
        n_rows=len(df),
        n_invalid_dates=n_invalid_dates,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Features temporelles et tranches horaires
# ---------------------------------------------------------------------------

def time_slot_label(dt: pd.Timestamp, slot_minutes: int = 60) -> str:
    """
    Construit le libelle d'une tranche horaire, ex: '2026-06-28_8h-9h'
    pour slot_minutes=60, ou '2026-06-28_8h00-8h30' pour slot_minutes=30.
    """
    minute_of_day = dt.hour * 60 + dt.minute
    slot_index = minute_of_day // slot_minutes
    start_minute = slot_index * slot_minutes
    end_minute = start_minute + slot_minutes
    start_h, start_m = divmod(start_minute, 60)
    end_h, end_m = divmod(end_minute % (24 * 60), 60)

    def fmt(h, m):
        return f"{h}h" if m == 0 else f"{h}h{m:02d}"

    return f"{dt.strftime('%Y-%m-%d')}_{fmt(start_h, start_m)}-{fmt(end_h, end_m)}"


def add_time_features(df: pd.DataFrame, slot_minutes: int = 60) -> pd.DataFrame:
    """
    Ajoute les colonnes temporelles necessaires a l'agregation :
    date, heure_debut (int, debut de tranche en heures), tranche (label),
    jour_semaine (nom FR), jour_semaine_num (0=Lundi), semaine, mois.

    Les lignes dont date_heure est invalide (NaT) sont supprimees.
    """
    out = df.copy()
    out = out.dropna(subset=["date_heure"])
    out["date"] = out["date_heure"].dt.date.astype(str)
    minute_of_day = out["date_heure"].dt.hour * 60 + out["date_heure"].dt.minute
    slot_index = (minute_of_day // slot_minutes).astype(int)
    out["heure_debut"] = (slot_index * slot_minutes) // 60
    out["minute_debut"] = (slot_index * slot_minutes) % 60
    out["tranche"] = out["date_heure"].apply(
        lambda d: time_slot_label(d, slot_minutes=slot_minutes)
    )
    out["jour_semaine_num"] = out["date_heure"].dt.weekday  # 0 = lundi
    out["jour_semaine"] = out["jour_semaine_num"].apply(lambda i: JOURS_SEMAINE_FR[i])
    out["semaine"] = out["date_heure"].dt.isocalendar().week.astype(int)
    out["mois"] = out["date_heure"].dt.month
    return out


def filter_holidays(
    df: pd.DataFrame,
    country: str = "FR",
    exclude: bool = True,
    custom_holidays: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Marque (colonne 'est_ferie') et, si exclude=True, retire les appels
    tombant un jour ferie.

    Parameters
    ----------
    country : code pays utilise par la librairie `holidays` (ex: "FR").
    custom_holidays : liste optionnelle de dates "YYYY-MM-DD" a traiter
        comme feries en plus de celles calculees automatiquement (utile
        pour des fermetures internes/ponts d'entreprise).
    """
    out = df.copy()
    years = pd.to_datetime(out["date"]).dt.year.unique().tolist()

    holiday_dates = set()
    if holidays_lib is not None and years:
        try:
            holiday_dates = set(
                d.isoformat() for d in holidays_lib.country_holidays(country, years=years)
            )
        except Exception:
            holiday_dates = set()

    if custom_holidays:
        holiday_dates |= set(custom_holidays)

    out["est_ferie"] = out["date"].isin(holiday_dates)

    if exclude:
        out = out[~out["est_ferie"]].drop(columns=["est_ferie"])

    return out


# ---------------------------------------------------------------------------
# Agregation par date + tranche horaire
# ---------------------------------------------------------------------------

def aggregate_by_slot(
    df: pd.DataFrame,
    wait_threshold_sec: float = 20.0,
) -> pd.DataFrame:
    """
    Agrege les appels par (date, tranche) et calcule les indicateurs
    necessaires a la prevision et au staffing.

    Le DataFrame en entree doit deja avoir passe `add_time_features`.

    Indicateurs calcules par groupe (date, tranche) :
        - volume               : nombre total d'appels
        - n_repondu / n_non_repondu
        - taux_decroche        : % d'appels non repondus OU repondus avec
                                  un temps d'attente > wait_threshold_sec
        - attente_moyenne      : temps d'attente moyen (secondes)
        - dmt_moyenne          : (post_appel + temps_communication +
                                  temps_attente) moyen, secondes
        - jour_semaine, jour_semaine_num, heure_debut, semaine, mois :
          reportes depuis le groupe (constants au sein d'une tranche)
    """
    work = df.copy()
    work["est_non_decroche"] = (work["statut"] == STATUT_NON_REPONDU) | (
        (work["statut"] == STATUT_REPONDU) & (work["temps_attente"] > wait_threshold_sec)
    )
    work["dmt_appel"] = (
        work["post_appel"].fillna(0)
        + work["temps_communication"].fillna(0)
        + work["temps_attente"].fillna(0)
    )

    group_cols = ["date", "tranche"]
    agg = work.groupby(group_cols).agg(
        volume=("id_appel", "count"),
        n_repondu=("statut", lambda s: (s == STATUT_REPONDU).sum()),
        n_non_repondu=("statut", lambda s: (s == STATUT_NON_REPONDU).sum()),
        n_non_decroche=("est_non_decroche", "sum"),
        attente_moyenne=("temps_attente", "mean"),
        dmt_moyenne=("dmt_appel", "mean"),
        jour_semaine=("jour_semaine", "first"),
        jour_semaine_num=("jour_semaine_num", "first"),
        heure_debut=("heure_debut", "first"),
        semaine=("semaine", "first"),
        mois=("mois", "first"),
    ).reset_index()

    agg["taux_decroche"] = agg["n_non_decroche"] / agg["volume"]
    agg = agg.drop(columns=["n_non_decroche"])
    agg["date"] = pd.to_datetime(agg["date"])
    agg = agg.sort_values(["date", "heure_debut"]).reset_index(drop=True)
    return agg


def dmt_by_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la DMT moyenne par agent et par date (pour suivre son
    evolution dans le temps, ex: amelioration progressive).
    Les appels non repondus (id_agent vide) sont exclus.
    """
    work = df[df["statut"] == STATUT_REPONDU].copy()
    work["dmt_appel"] = (
        work["post_appel"].fillna(0)
        + work["temps_communication"].fillna(0)
        + work["temps_attente"].fillna(0)
    )
    out = work.groupby(["id_agent", "date"]).agg(
        dmt_moyenne=("dmt_appel", "mean"),
        n_appels=("id_appel", "count"),
    ).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["id_agent", "date"]).reset_index(drop=True)


def filter_by_agent_and_date(
    df: pd.DataFrame,
    agent: Optional[str] = None,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> pd.DataFrame:
    """Filtre un DataFrame d'appels (ou de DMT par agent) par agent et plage de dates."""
    out = df.copy()
    if agent and agent != "Tous" and "id_agent" in out.columns:
        out = out[out["id_agent"] == agent]
    if "date" in out.columns:
        if date_min:
            out = out[pd.to_datetime(out["date"]) >= pd.to_datetime(date_min)]
        if date_max:
            out = out[pd.to_datetime(out["date"]) <= pd.to_datetime(date_max)]
    return out


# ---------------------------------------------------------------------------
# Validation de la distribution de Poisson
# ---------------------------------------------------------------------------

def lambda_table(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule, pour chaque (jour_semaine, heure_debut), la moyenne (lambda)
    et la variance du volume d'appels observe sur l'historique, ainsi
    que le nombre d'observations (n_dates).

    Pour une vraie distribution de Poisson, moyenne ~ variance : on
    calcule donc aussi le ratio variance/moyenne (proche de 1 = bon signe,
    largement > 1 = sur-dispersion frequente avec des appels en pratique).
    """
    table = agg.groupby(["jour_semaine", "jour_semaine_num", "heure_debut"]).agg(
        lambda_=("volume", "mean"),
        variance=("volume", "var"),
        n_observations=("volume", "count"),
    ).reset_index()
    table["ratio_var_moyenne"] = table["variance"] / table["lambda_"]
    table = table.sort_values(["jour_semaine_num", "heure_debut"]).drop(
        columns=["jour_semaine_num"]
    )
    return table.reset_index(drop=True)


def poisson_chi2_test(observed_counts: np.ndarray, lam: float, n_bins: int = 8):
    """
    Test du Chi2 d'adequation entre les comptages observes et une loi de
    Poisson(lam). Regroupe les valeurs en classes (bins) pour respecter
    les conditions de validite du test (effectifs attendus suffisants).

    Returns
    -------
    dict avec 'chi2_stat', 'p_value', 'ddl', 'valide' (bool : test jugeable),
    et un message d'interpretation en francais.
    """
    observed_counts = np.asarray(observed_counts)
    observed_counts = observed_counts[~np.isnan(observed_counts)]
    n = len(observed_counts)

    if n < 5 or lam <= 0:
        return {
            "chi2_stat": np.nan,
            "p_value": np.nan,
            "ddl": 0,
            "valide": False,
            "message": "Pas assez d'observations pour realiser un test fiable.",
        }

    max_k = int(max(observed_counts.max(), stats.poisson.ppf(0.999, lam)))
    edges = np.unique(
        np.clip(np.linspace(0, max_k + 1, num=min(n_bins, max_k + 2) + 1), 0, max_k + 1)
    ).astype(int)
    if len(edges) < 3:
        edges = np.array([0, max_k // 2 + 1, max_k + 1])

    observed_hist, _ = np.histogram(observed_counts, bins=edges)
    expected_probs = np.diff(stats.poisson.cdf(edges - 1, lam))
    expected_probs[-1] += 1 - expected_probs.sum()  # capter la queue de distribution
    expected_counts = expected_probs * n

    # Fusionner les classes dont l'effectif attendu est trop faible (<5)
    obs_merged, exp_merged = [], []
    obs_buf, exp_buf = 0.0, 0.0
    for o, e in zip(observed_hist, expected_counts):
        obs_buf += o
        exp_buf += e
        if exp_buf >= 5:
            obs_merged.append(obs_buf)
            exp_merged.append(exp_buf)
            obs_buf, exp_buf = 0.0, 0.0
    if exp_buf > 0:
        if exp_merged:
            obs_merged[-1] += obs_buf
            exp_merged[-1] += exp_buf
        else:
            obs_merged.append(obs_buf)
            exp_merged.append(exp_buf)

    if len(obs_merged) < 2:
        return {
            "chi2_stat": np.nan,
            "p_value": np.nan,
            "ddl": 0,
            "valide": False,
            "message": "Pas assez de classes avec un effectif suffisant pour ce test.",
        }

    obs_merged = np.array(obs_merged)
    exp_merged = np.array(exp_merged) * (obs_merged.sum() / np.array(exp_merged).sum())
    ddl = max(len(obs_merged) - 1 - 1, 1)  # -1 pour le total, -1 car lambda estime
    chi2_stat, p_value = stats.chisquare(obs_merged, exp_merged, ddof=max(ddl - (len(obs_merged) - 1), 0))

    if p_value > 0.05:
        message = (
            f"p-value = {p_value:.3f} (> 0.05) : on ne peut PAS rejeter "
            "l'hypothese de Poisson. Vos donnees sont compatibles avec "
            f"une loi de Poisson(lambda={lam:.1f})."
        )
    else:
        message = (
            f"p-value = {p_value:.3f} (<= 0.05) : l'hypothese de Poisson "
            "est rejetee pour ce groupe. La variance observee s'ecarte "
            "trop de la moyenne (souvent un signe de sur-dispersion)."
        )

    return {
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "ddl": int(ddl),
        "valide": True,
        "message": message,
    }


def poisson_validation_report(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un rapport complet (1 ligne par jour_semaine x heure_debut) :
    lambda, variance, ratio, et resultat du test du Chi2.

    Retourne un DataFrame pret a etre affiche dans Streamlit.
    """
    lam_table = lambda_table(agg)
    rows = []
    for _, row in lam_table.iterrows():
        mask = (agg["jour_semaine"] == row["jour_semaine"]) & (
            agg["heure_debut"] == row["heure_debut"]
        )
        observed = agg.loc[mask, "volume"].values
        test_result = poisson_chi2_test(observed, row["lambda_"])
        rows.append({
            "jour_semaine": row["jour_semaine"],
            "heure_debut": row["heure_debut"],
            "lambda": round(row["lambda_"], 2),
            "variance": round(row["variance"], 2) if pd.notna(row["variance"]) else np.nan,
            "ratio_var_moyenne": round(row["ratio_var_moyenne"], 2) if pd.notna(row["ratio_var_moyenne"]) else np.nan,
            "n_observations": int(row["n_observations"]),
            "p_value": round(test_result["p_value"], 4) if test_result["valide"] else np.nan,
            "conclusion": test_result["message"],
        })
    return pd.DataFrame(rows)


def poisson_pmf_curve(lam: float, k_max: Optional[int] = None) -> pd.DataFrame:
    """Retourne les points (k, P(X=k)) de la loi de Poisson(lam) pour le trace graphique."""
    if k_max is None:
        k_max = max(int(stats.poisson.ppf(0.999, lam)) + 1, 5)
    ks = np.arange(0, k_max + 1)
    probs = stats.poisson.pmf(ks, lam)
    return pd.DataFrame({"k": ks, "proba": probs})


# ---------------------------------------------------------------------------
# Construction du squelette de previsions futures
# ---------------------------------------------------------------------------

def build_future_skeleton(
    last_date: pd.Timestamp,
    horizon_days: int,
    slots: Sequence[int],
    country: str = "FR",
    exclude_holidays: bool = True,
    include_weekends: bool = True,
    custom_holidays: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Construit le calendrier (date x tranche horaire) a prevoir pour les
    `horizon_days` jours suivant `last_date`.

    Returns
    -------
    pd.DataFrame avec les colonnes : date, heure_debut, tranche,
    jour_semaine, jour_semaine_num (memes conventions que `add_time_features`
    / `aggregate_by_slot`, ce qui permet de reutiliser directement les
    modeles de `models.py`).
    """
    future_dates = pd.date_range(
        pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=horizon_days, freq="D"
    )

    if not include_weekends:
        future_dates = future_dates[future_dates.weekday < 5]

    if exclude_holidays:
        years = future_dates.year.unique().tolist()
        holiday_dates = set()
        if holidays_lib is not None and years:
            try:
                holiday_dates = set(
                    d.isoformat() for d in holidays_lib.country_holidays(country, years=years)
                )
            except Exception:
                holiday_dates = set()
        if custom_holidays:
            holiday_dates |= set(custom_holidays)
        future_dates = [d for d in future_dates if d.strftime("%Y-%m-%d") not in holiday_dates]

    rows = []
    for d in future_dates:
        for slot in slots:
            rows.append({
                "date": pd.Timestamp(d),
                "heure_debut": slot,
                "tranche": f"{d.strftime('%Y-%m-%d')}_{slot}h-{slot + 1}h",
                "jour_semaine": JOURS_SEMAINE_FR[d.weekday()],
                "jour_semaine_num": d.weekday(),
            })
    return pd.DataFrame(rows)
