"""
utils.py
========
Fonctions de prétraitement, d'agrégation et de validation statistique
pour l'application de prévision des appels et de recommandation de staffing.

Les fonctions sont indépendantes de Streamlit et testées dans tests/.

Colonnes attendues dans le CSV d'entrée (voir README.md) :
    - id_appel             : identifiant unique de l'appel
    - date_heure           : date et heure de l'appel (parsable par pandas)
    - statut               : "repondu" ou "non_repondu"
    - post_appel           : temps de post-appel, en secondes
    - temps_communication  : durée de la conversation, en secondes
    - temps_attente        : temps d'attente avant prise d'appel, en secondes
    - id_agent             : identifiant de l'agent ayant traité l'appel
                              (peut être vide si statut == "non_repondu")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

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
    """Résultat de la validation d'un fichier CSV importé."""
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
        Données brutes, avec date_heure converti en datetime.
    """
    df = pd.read_csv(file_or_buffer)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date_heure" in df.columns:
        df["date_heure"] = pd.to_datetime(df["date_heure"], errors="coerce")
    return df


def validate_calls_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Vérifie que le DataFrame contient les colonnes requises et des dates
    valides. Ne lève pas d'exception : retourne un rapport structuré pour
    affichage côté interface (utile pour un public non technique).
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

def hour_label(heure: float) -> str:
    """Libellé court d'une heure de début : 8 -> '8h', 8.5 -> '8h30'."""
    h = int(heure)
    minutes = int(round((heure - h) * 60))
    return f"{h}h" if minutes == 0 else f"{h}h{minutes:02d}"


def time_slot_label(dt: pd.Timestamp, slot_minutes: int = 60) -> str:
    """
    Construit le libellé d'une tranche horaire, ex: '2026-06-28_8h-9h'
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
    Ajoute les colonnes temporelles nécessaires a l'agrégation :
    date, heure_début (début de tranche en heures, 8.5 = 8h30), tranche (label),
    jour_semaine (nom FR), jour_semaine_num (0=Lundi), semaine, mois.

    Les lignes dont date_heure est invalide (NaT) sont supprimées.
    """
    out = df.copy()
    out = out.dropna(subset=["date_heure"])
    out["date"] = out["date_heure"].dt.date.astype(str)
    minute_of_day = out["date_heure"].dt.hour * 60 + out["date_heure"].dt.minute
    slot_index = (minute_of_day // slot_minutes).astype(int)
    out["heure_debut"] = (slot_index * slot_minutes) / 60  # 8.5 = 8h30
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
    Marque (colonne 'est_férié') et, si exclude=True, retire les appels
    tombant un jour férié.

    Parameters
    ----------
    country : code pays utilise par la librairie `holidays` (ex: "FR").
    custom_holidays : liste optionnelle de dates "YYYY-MM-DD" a traiter
        comme fériés en plus de celles calculées automatiquement (utile
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
# Agrégation par date + tranche horaire
# ---------------------------------------------------------------------------

def aggregate_by_slot(
    df: pd.DataFrame,
    wait_threshold_sec: float = 20.0,
    slot_minutes: int = 60,
) -> pd.DataFrame:
    """
    Agrège les appels par (date, tranche). Le DataFrame doit être passé par
    `add_time_features`.

    Chaque tranche d'ouverture apparaît, même sans aucun appel (volume = 0) :
    retirer ces zéros biaiserait les moyennes vers le haut.

    Indicateurs par tranche :
        - volume : appels reçus
        - n_repondu, n_abandonne, n_dans_delai
        - taux_abandon : part des appels reçus puis abandonnés
        - niveau_service : part des appels reçus, pris en moins de
          `wait_threshold_sec` secondes
        - attente_moyenne : attente moyenne des appels répondus (secondes)
        - dmt_moyenne : temps de communication + post-appel des appels répondus.
          L'attente n'en fait pas partie, car un agent ne travaille pas pendant
          qu'un appelant patiente.
    """
    columns = [
        "date", "tranche", "heure_debut", "jour_semaine", "jour_semaine_num", "semaine", "mois",
        "volume", "n_repondu", "n_abandonne", "n_dans_delai", "taux_abandon", "niveau_service",
        "attente_moyenne", "dmt_moyenne",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    repondu = work["statut"] == STATUT_REPONDU
    work["_repondu"] = repondu
    work["_abandonne"] = work["statut"] == STATUT_NON_REPONDU
    work["_dans_delai"] = repondu & (work["temps_attente"] <= wait_threshold_sec)
    work["_attente"] = work["temps_attente"].where(repondu)
    work["_dmt"] = (work["post_appel"].fillna(0) + work["temps_communication"].fillna(0)).where(repondu)

    observed = work.groupby(["date", "heure_debut"]).agg(
        volume=("id_appel", "count"),
        n_repondu=("_repondu", "sum"),
        n_abandonne=("_abandonne", "sum"),
        n_dans_delai=("_dans_delai", "sum"),
        attente_moyenne=("_attente", "mean"),
        dmt_moyenne=("_dmt", "mean"),
    )

    grid = pd.MultiIndex.from_product(
        [sorted(work["date"].unique()), sorted(work["heure_debut"].unique())],
        names=["date", "heure_debut"],
    )
    agg = observed.reindex(grid).reset_index()
    counts = ["volume", "n_repondu", "n_abandonne", "n_dans_delai"]
    agg[counts] = agg[counts].fillna(0).astype(int)

    agg["date"] = pd.to_datetime(agg["date"])
    agg["jour_semaine_num"] = agg["date"].dt.weekday
    agg["jour_semaine"] = agg["jour_semaine_num"].map(lambda i: JOURS_SEMAINE_FR[i])
    agg["semaine"] = agg["date"].dt.isocalendar().week.astype(int)
    agg["mois"] = agg["date"].dt.month
    starts = agg["date"] + pd.to_timedelta(agg["heure_debut"], unit="h")
    agg["tranche"] = [time_slot_label(start, slot_minutes) for start in starts]

    received = agg["volume"].where(agg["volume"] > 0)
    agg["taux_abandon"] = agg["n_abandonne"] / received
    agg["niveau_service"] = agg["n_dans_delai"] / received
    return agg[columns].sort_values(["date", "heure_debut"]).reset_index(drop=True)


def kpi_summary(agg: pd.DataFrame) -> dict:
    """KPI globaux, pondérés par les volumes (pas une moyenne de moyennes)."""
    volume = agg["volume"].sum()
    answered = agg["n_repondu"].sum()
    if volume == 0:
        return {"volume": 0, "taux_abandon": np.nan, "niveau_service": np.nan,
                "dmt_moyenne": np.nan, "attente_moyenne": np.nan}
    return {
        "volume": int(volume),
        "taux_abandon": agg["n_abandonne"].sum() / volume,
        "niveau_service": agg["n_dans_delai"].sum() / volume,
        "dmt_moyenne": (agg["dmt_moyenne"] * agg["n_repondu"]).sum() / answered if answered else np.nan,
        "attente_moyenne": (agg["attente_moyenne"] * agg["n_repondu"]).sum() / answered if answered else np.nan,
    }


def dmt_by_agent(df: pd.DataFrame) -> pd.DataFrame:
    """DMT moyenne par agent et par date, appels répondus uniquement."""
    work = df[df["statut"] == STATUT_REPONDU].copy()
    work["dmt_appel"] = work["post_appel"].fillna(0) + work["temps_communication"].fillna(0)
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
    Moyenne (lambda), variance et nombre d'observations du volume pour chaque
    (jour_semaine, heure_début). Sous une loi de Poisson, le ratio
    variance/moyenne vaut environ 1 ; au-dessus, on parle de sur-dispersion.
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


def dispersion_test(counts: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Test de dispersion : sous une loi de Poisson, (n - 1) * variance / moyenne
    suit un Chi2 à n - 1 degrés de liberte. Le test est unilatéral, car c'est la
    sur-dispersion qui ferait sous-estimer les pics d'appels.

    Sur une seule tranche, il n'y a qu'une douzaine d'observations : ne pas
    rejeter ne prouve donc rien. `pooled_dispersion_test` est bien plus puissant.
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[~np.isnan(counts)]
    n, mean = len(counts), counts.mean() if len(counts) else 0.0

    if n < 5 or mean <= 0:
        return {
            "ratio": np.nan, "p_value": np.nan, "valide": False,
            "message": "Pas assez d'observations pour réaliser un test fiable.",
        }

    ratio = counts.var(ddof=1) / mean
    p_value = float(stats.chi2.sf((n - 1) * ratio, n - 1))
    if p_value < alpha:
        message = (
            f"p-value = {p_value:.3f} : sur-dispersion détectée (variance = {ratio:.2f} × moyenne). "
            "Les pics sont plus forts que ce que prévoit une loi de Poisson."
        )
    else:
        message = (
            f"p-value = {p_value:.3f} : pas de sur-dispersion détectée sur cette tranche, "
            f"mais avec {n} observations seulement, ce test isolé a peu de puissance."
        )
    return {"ratio": float(ratio), "p_value": p_value, "valide": True, "message": message}


def pooled_dispersion_test(agg: pd.DataFrame, min_lambda: float = 1.0, alpha: float = 0.05) -> dict:
    """
    Test de dispersion regroupant toutes les tranches (jour x heure) : chacune
    garde sa propre moyenne, mais leurs écarts sont additionnés. C'est le test à
    regarder pour décider si le modèle de Poisson convient.
    """
    grouped = agg.groupby(["jour_semaine_num", "heure_debut"])["volume"]
    stat, ddl, n_groupes = 0.0, 0, 0
    for _, counts in grouped:
        mean = counts.mean()
        if len(counts) < 2 or mean < min_lambda:
            continue
        stat += ((counts - mean) ** 2).sum() / mean
        ddl += len(counts) - 1
        n_groupes += 1

    if ddl == 0:
        return {
            "ratio": np.nan, "p_value": np.nan, "n_groupes": 0, "valide": False,
            "message": "Pas assez d'observations pour réaliser le test global.",
        }

    ratio = stat / ddl
    p_value = float(stats.chi2.sf(stat, ddl))
    if p_value < alpha:
        message = (
            f"Sur-dispersion détectée (ratio {ratio:.2f}, p-value = {p_value:.4f}) : la variance "
            "dépasse la moyenne, les intervalles de Poisson seraient trop étroits."
        )
    else:
        message = (
            f"Aucune sur-dispersion détectée sur {n_groupes} tranches "
            f"(ratio {ratio:.2f}, p-value = {p_value:.3f}) : le modèle de Poisson est adapté."
        )
    return {"ratio": float(ratio), "p_value": p_value, "n_groupes": n_groupes,
            "valide": True, "message": message}


def poisson_validation_report(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Rapport par (jour_semaine, heure_début). `p_ajustee` corrige les comparaisons
    multiples (Benjamini-Hochberg) : sur 77 tranches, environ 4 auraient sinon
    une p-value < 0.05 par simple hasard.
    """
    rows = []
    for _, row in lambda_table(agg).iterrows():
        mask = (agg["jour_semaine"] == row["jour_semaine"]) & (agg["heure_debut"] == row["heure_debut"])
        result = dispersion_test(agg.loc[mask, "volume"].values)
        rows.append({
            "jour_semaine": row["jour_semaine"],
            "heure_debut": row["heure_debut"],
            "lambda": round(row["lambda_"], 2),
            "variance": round(row["variance"], 2) if pd.notna(row["variance"]) else np.nan,
            "ratio_var_moyenne": round(row["ratio_var_moyenne"], 2) if pd.notna(row["ratio_var_moyenne"]) else np.nan,
            "n_observations": int(row["n_observations"]),
            "p_value": result["p_value"],
        })
    report = pd.DataFrame(rows)
    report["p_ajustee"] = np.nan
    valid = report["p_value"].notna()
    if valid.any():
        report.loc[valid, "p_ajustee"] = multipletests(report.loc[valid, "p_value"], method="fdr_bh")[1]
    return report.round({"p_value": 4, "p_ajustee": 4})


def poisson_pmf_curve(lam: float, k_max: Optional[int] = None) -> pd.DataFrame:
    """Retourne les points (k, P(X=k)) de la loi de Poisson(lam) pour le tracé graphique."""
    if k_max is None:
        k_max = max(int(stats.poisson.ppf(0.999, lam)) + 1, 5)
    ks = np.arange(0, k_max + 1)
    probs = stats.poisson.pmf(ks, lam)
    return pd.DataFrame({"k": ks, "proba": probs})


# ---------------------------------------------------------------------------
# Construction du squelette de prévisions futures
# ---------------------------------------------------------------------------

def build_future_skeleton(
    last_date: pd.Timestamp,
    horizon_days: int,
    slots: Sequence[int],
    country: str = "FR",
    exclude_holidays: bool = True,
    include_weekends: bool = True,
    custom_holidays: Optional[Sequence[str]] = None,
    slot_minutes: int = 60,
) -> pd.DataFrame:
    """
    Construit le calendrier (date x tranche horaire) à prévoir pour les
    `horizon_days` jours suivant `last_date`.

    Returns
    -------
    pd.DataFrame avec les colonnes : date, heure_début, tranche,
    jour_semaine, jour_semaine_num (mêmes conventions que `add_time_features`
    / `aggregate_by_slot`, ce qui permet de réutiliser directement les
    modèles de `models.py`).
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
                "tranche": time_slot_label(pd.Timestamp(d) + pd.Timedelta(hours=float(slot)), slot_minutes),
                "jour_semaine": JOURS_SEMAINE_FR[d.weekday()],
                "jour_semaine_num": d.weekday(),
            })
    return pd.DataFrame(rows)
