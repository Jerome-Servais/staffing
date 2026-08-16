"""
app.py
======
Application Streamlit : prevision du volume d'appels entrants et
recommandation du nombre d'agents necessaires (staffing), par date et
tranche horaire.

Lancement :
    streamlit run app.py

Structure (voir README.md pour le detail) :
    - Exploration des donnees
    - Validation de la distribution de Poisson
    - Prediction et comparaison de modeles
    - Recommandation de staffing
    - Donnees synthetiques (demo)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate_synthetic_calls
from models import (
    MODEL_REGISTRY,
    estimate_future_dmt_and_quality,
    evaluate_model,
    recommend_staffing,
)
from utils import (
    JOURS_SEMAINE_FR,
    add_time_features,
    aggregate_by_slot,
    build_future_skeleton,
    dmt_by_agent,
    filter_by_agent_and_date,
    filter_holidays,
    load_calls_csv,
    poisson_chi2_test,
    poisson_pmf_curve,
    poisson_validation_report,
    validate_calls_dataframe,
)


# ---------------------------------------------------------------------------
# Pipeline de pretraitement (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Pretraitement des donnees...")
def process_pipeline(raw_df: pd.DataFrame, slot_minutes: int, wait_threshold_sec: float,
                      exclude_holidays: bool, country: str = "FR"):
    feat = add_time_features(raw_df, slot_minutes=slot_minutes)
    feat = filter_holidays(feat, country=country, exclude=exclude_holidays)
    agg = aggregate_by_slot(feat, wait_threshold_sec=wait_threshold_sec)
    dmt_agent = dmt_by_agent(feat)
    return feat, agg, dmt_agent


# ---------------------------------------------------------------------------
# Page 1 : Exploration des donnees
# ---------------------------------------------------------------------------

def page_exploration(feat: pd.DataFrame, agg: pd.DataFrame, dmt_agent: pd.DataFrame):
    st.header("🔍 Exploration des données")

    if agg.empty:
        st.warning("Aucune donnée disponible apres filtrage. Verifiez votre fichier ou vos parametres.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Appels totaux", f"{int(agg['volume'].sum()):,}".replace(",", " "))
    col2.metric("Taux de décroché moyen", f"{agg['taux_decroche'].mean() * 100:.1f} %")
    col3.metric("DMT moyenne", f"{agg['dmt_moyenne'].mean():.0f} s")
    col4.metric("Attente moyenne", f"{agg['attente_moyenne'].mean():.1f} s")

    st.subheader("Volume d'appels par date et tranche horaire")
    pivot = agg.pivot_table(index="heure_debut", columns="date", values="volume", aggfunc="sum")
    fig = px.imshow(
        pivot,
        labels=dict(x="Date", y="Heure de début", color="Volume"),
        aspect="auto",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("DMT moyenne par agent au fil du temps")
    agents = sorted(dmt_agent["id_agent"].dropna().unique())
    if agents:
        default_agents = agents[:5] if len(agents) > 5 else agents
        selected_agent = st.selectbox("Filtrer par agent", ["Tous"] + agents)
        date_min, date_max = agg["date"].min(), agg["date"].max()
        plot_data = filter_by_agent_and_date(dmt_agent, agent=selected_agent)
        if selected_agent == "Tous":
            plot_data = plot_data[plot_data["id_agent"].isin(default_agents)]
            st.caption(f"Affichage des {len(default_agents)} premiers agents (filtrez ci-dessus pour en voir un seul).")
        fig2 = px.line(
            plot_data, x="date", y="dmt_moyenne", color="id_agent", markers=True,
            labels={"dmt_moyenne": "DMT moyenne (s)", "date": "Date", "id_agent": "Agent"},
        )
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("Aucun agent identifie dans les donnees (tous les appels sont peut-etre non repondus).")

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Temps d'attente moyen par tranche")
        by_slot = agg.groupby("heure_debut")["attente_moyenne"].mean().reset_index()
        fig3 = px.bar(
            by_slot, x="heure_debut", y="attente_moyenne",
            labels={"heure_debut": "Heure de début", "attente_moyenne": "Attente moyenne (s)"},
        )
        st.plotly_chart(fig3, width='stretch')
    with col6:
        st.subheader("Taux de décroché par jour de la semaine")
        by_day = (
            agg.groupby("jour_semaine")["taux_decroche"].mean()
            .reindex(JOURS_SEMAINE_FR).dropna().reset_index()
        )
        fig4 = px.bar(
            by_day, x="jour_semaine", y="taux_decroche",
            labels={"jour_semaine": "Jour", "taux_decroche": "Taux de décroché"},
        )
        fig4.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig4, width='stretch')

    st.subheader("Données agrégées (date + tranche)")
    st.dataframe(agg, width='stretch', hide_index=True)
    csv_bytes = agg.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger les données agrégées (CSV)", csv_bytes, "donnees_agregees.csv", "text/csv")


# ---------------------------------------------------------------------------
# Page 2 : Validation de Poisson
# ---------------------------------------------------------------------------

def page_poisson(agg: pd.DataFrame):
    st.header("📐 Validation de la distribution de Poisson")

    with st.expander("ℹ️ Pourquoi vérifier une distribution de Poisson ? (cliquez pour lire)", expanded=True):
        st.markdown(
            """
Une **loi de Poisson** est une façon de modéliser des événements aléatoires
et indépendants (comme des appels entrants) qui surviennent à un rythme
moyen à peu près constant sur une période donnée, qu'on appelle **lambda
(λ)**.

**Exemple concret :** si en moyenne vous recevez 50 appels entre 10h et 11h
un lundi, alors λ = 50. La loi de Poisson permet de calculer, par exemple,
la probabilité de recevoir exactement 48 appels un lundi donné entre 10h et
11h.

**Pourquoi c'est utile ?** Si vos données suivent bien une loi de Poisson,
cela justifie l'utilisation de modèles de prévision adaptés aux comptages
(régression de Poisson, Poisson bayésien...), qui sont en général plus
fiables qu'une simple régression linéaire pour ce type de données.

**Comment le vérifier ?**
- **Règle pratique :** pour une vraie loi de Poisson, la *moyenne* et la
  *variance* du nombre d'appels doivent être proches.
- **Test du Chi²:** compare les fréquences observées aux fréquences
  attendues sous une loi de Poisson. Une **p-value > 0.05** signifie que
  vos données sont compatibles avec une loi de Poisson (on ne peut pas
  prouver le contraire) ; une p-value ≤ 0.05 suggère un écart significatif
  (souvent une variance plus grande que la moyenne, par exemple lors de
  pics d'appels imprévisibles).
            """
        )

    if agg.empty:
        st.warning("Aucune donnée disponible apres filtrage.")
        return

    jours_disponibles = [j for j in JOURS_SEMAINE_FR if j in agg["jour_semaine"].unique()]
    if not jours_disponibles:
        st.warning("Aucun jour disponible dans les donnees.")
        return

    jour = st.selectbox("Jour de la semaine", jours_disponibles)
    heures = sorted(agg.loc[agg["jour_semaine"] == jour, "heure_debut"].unique())
    heure = st.selectbox("Tranche horaire à visualiser", heures, format_func=lambda h: f"{h}h-{h + 1}h")

    obs = agg[(agg["jour_semaine"] == jour) & (agg["heure_debut"] == heure)]["volume"].dropna()

    if len(obs) < 3:
        st.warning("Pas assez d'observations pour ce jour/tranche.")
    else:
        lam = float(obs.mean())
        variance = float(obs.var())
        st.markdown(
            f"**λ (moyenne) = {lam:.2f}** &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**variance = {variance:.2f}** &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**ratio variance/moyenne = {variance / lam if lam else float('nan'):.2f}** "
            f"(≈ 1 est un bon signe pour Poisson)"
        )

        k_max = int(obs.max()) + 2
        counts, edges = np.histogram(obs, bins=np.arange(0, k_max + 2))
        hist_df = pd.DataFrame({"k": edges[:-1], "observe": counts})
        poisson_df = poisson_pmf_curve(lam, k_max=k_max)
        poisson_df["attendu"] = poisson_df["proba"] * len(obs)

        fig = go.Figure()
        fig.add_bar(x=hist_df["k"], y=hist_df["observe"], name="Observé (historique)")
        fig.add_scatter(
            x=poisson_df["k"], y=poisson_df["attendu"], name=f"Théorique Poisson(λ={lam:.1f})",
            mode="lines+markers", line=dict(color="firebrick"),
        )
        fig.update_layout(
            xaxis_title="Nombre d'appels dans la tranche", yaxis_title="Nombre de jours observés",
            title=f"{jour}, {heure}h-{heure + 1}h",
        )
        st.plotly_chart(fig, width='stretch')

        test_result = poisson_chi2_test(obs.values, lam)
        if not test_result["valide"]:
            st.info(test_result["message"])
        elif test_result["p_value"] > 0.05:
            st.success(test_result["message"])
        else:
            st.warning(test_result["message"])

    st.markdown("### Tableau récapitulatif — tous les jours et tranches horaires")
    full_report = poisson_validation_report(agg)
    st.dataframe(
        full_report.drop(columns=["conclusion"]), width='stretch', hide_index=True
    )
    st.caption(
        "La colonne 'p_value' est vide quand il n'y a pas assez d'observations pour réaliser le "
        "test de manière fiable. Une p_value > 0.05 indique une bonne compatibilité avec une loi de Poisson."
    )


# ---------------------------------------------------------------------------
# Page 3 : Prediction et comparaison de modeles
# ---------------------------------------------------------------------------

def page_prediction(agg: pd.DataFrame):
    st.header("📈 Prévision et comparaison de modèles")

    if agg.empty or agg["date"].nunique() < 14:
        st.warning(
            "Historique insuffisant (moins de 2 semaines de données). Les prévisions seront peu fiables."
        )

    with st.expander("ℹ️ Quels modèles, et pourquoi ?"):
        for name, cls in MODEL_REGISTRY.items():
            st.markdown(f"**{name}**")
            st.caption(cls.description)

    model_names = list(MODEL_REGISTRY.keys())
    selected = st.multiselect("Modèles à comparer", model_names, default=model_names)
    test_days = st.slider("Nombre de jours réservés au test (backtest)", 7, 60, 14)

    if st.button("🚀 Lancer la comparaison", type="primary") and selected:
        max_date = agg["date"].max()
        split_date = max_date - pd.Timedelta(days=test_days)
        train = agg[agg["date"] <= split_date].copy()
        test = agg[agg["date"] > split_date].copy()

        if len(train) < 14 or len(test) < 1:
            st.error("Pas assez de données pour ce découpage train/test. Réduisez le nombre de jours de test.")
        else:
            results = {}
            progress = st.progress(0.0)
            for i, name in enumerate(selected):
                try:
                    with st.spinner(f"Entraînement : {name}..."):
                        model = MODEL_REGISTRY[name]()
                        res = evaluate_model(model, train, test)
                    results[name] = res
                except Exception as exc:  # pragma: no cover
                    st.warning(f"{name} : échec de l'entraînement ({exc})")
                progress.progress((i + 1) / len(selected))
            progress.empty()
            st.session_state["backtest_results"] = results
            st.session_state["backtest_test_df"] = test.reset_index(drop=True)

    results = st.session_state.get("backtest_results")
    test_df = st.session_state.get("backtest_test_df")

    if results:
        rows = [{"Modèle": name, **res["metrics"]} for name, res in results.items()]
        metrics_df = pd.DataFrame(rows).set_index("Modèle")
        st.subheader("Comparaison des métriques")
        st.dataframe(metrics_df.style.format("{:.2f}"), width='stretch')

        with st.expander("ℹ️ Comment interpréter ces métriques ?"):
            st.markdown(
                """
- **RMSE / MAE** : erreur moyenne, en nombre d'appels (plus petit = mieux).
  Le RMSE pénalise davantage les grosses erreurs que le MAE.
- **MAPE (%)** : erreur moyenne en pourcentage du volume réel (plus petit = mieux).
- **R²** : part de la variation du volume expliquée par le modèle (proche de 1 = mieux).
- **Couverture IC95% (%)** *(modèles probabilistes uniquement)* : pourcentage
  des vraies valeurs effectivement comprises dans l'intervalle prédit à 95%.
  On vise environ 95%. Trop bas = intervalle trop étroit (l'incertitude est
  sous-estimée) ; trop haut (proche de 100%) = intervalle probablement trop large.
- **Largeur moyenne IC** : largeur moyenne de l'intervalle à 95%. À couverture
  égale, un intervalle plus étroit est préférable (plus informatif).
                """
            )

        valid_rmse = metrics_df["RMSE"].dropna()
        if not valid_rmse.empty:
            best_model = valid_rmse.idxmin()
            st.success(f"🏆 Meilleur modèle sur ce backtest (RMSE le plus faible) : **{best_model}**")

            st.subheader("Prédictions vs données réelles")
            model_choices = list(results.keys())
            chosen = st.selectbox(
                "Modèle à visualiser", model_choices, index=model_choices.index(best_model)
            )
            preds = results[chosen]["predictions"].reset_index(drop=True)
            plot_df = test_df.copy()
            plot_df["pred_mean"] = preds["pred_mean"]
            plot_df["pred_lower"] = preds["pred_lower"]
            plot_df["pred_upper"] = preds["pred_upper"]

            by_date = plot_df.groupby("date").agg(
                volume=("volume", "sum"),
                pred_mean=("pred_mean", "sum"),
                pred_lower=("pred_lower", "sum"),
                pred_upper=("pred_upper", "sum"),
            ).reset_index()

            fig = go.Figure()
            if results[chosen]["has_interval"]:
                fig.add_scatter(x=by_date["date"], y=by_date["pred_upper"], line=dict(width=0), showlegend=False)
                fig.add_scatter(
                    x=by_date["date"], y=by_date["pred_lower"], fill="tonexty",
                    fillcolor="rgba(99,110,250,0.2)", line=dict(width=0), name="Intervalle 95%",
                )
            fig.add_scatter(x=by_date["date"], y=by_date["volume"], mode="lines+markers", name="Réel")
            fig.add_scatter(x=by_date["date"], y=by_date["pred_mean"], mode="lines+markers", name="Prédit")
            fig.update_layout(xaxis_title="Date", yaxis_title="Volume d'appels (toutes tranches confondues)")
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Les volumes sont additionnés sur toutes les tranches horaires de chaque jour pour une "
                "lecture simplifiée. Le détail par tranche est disponible dans le tableau ci-dessous."
            )
            st.dataframe(
                plot_df[["date", "tranche", "heure_debut", "volume", "pred_mean", "pred_lower", "pred_upper"]],
                width='stretch', hide_index=True,
            )
    else:
        st.info("Sélectionnez des modèles puis cliquez sur « Lancer la comparaison ».")


# ---------------------------------------------------------------------------
# Page 4 : Recommandation de staffing
# ---------------------------------------------------------------------------

def page_staffing(agg: pd.DataFrame, slot_duration_sec: float):
    st.header("👥 Recommandation de staffing")

    if agg.empty:
        st.warning("Aucune donnée disponible apres filtrage.")
        return

    with st.expander("ℹ️ Comment est calculée la recommandation ?"):
        st.markdown(
            """
1. **Besoin de base** : `agents = CEIL( (volume prédit × DMT moyenne) / durée de la tranche )`
   — le nombre d'agents nécessaire pour écouler la charge de travail totale.
2. **Ajustement qualité de service** : si l'attente ou le taux de décroché
   estimés dépassent vos seuils, le nombre d'agents est augmenté
   progressivement (en supposant, en première approximation, que l'attente
   et le taux de décroché diminuent proportionnellement au nombre d'agents
   ajoutés), jusqu'à respecter les deux seuils ou atteindre le maximum
   d'agents disponibles.

*Note : il s'agit d'une approximation simple. Un modèle de file d'attente
(loi d'Erlang C) donnerait une estimation plus précise de l'effet du nombre
d'agents sur l'attente, mais sort du cadre de ce prototype.*
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox("Modèle de prévision", list(MODEL_REGISTRY.keys()))
        horizon = st.slider("Horizon de prévision (jours)", 1, 30, 7)
    with col2:
        include_weekends = st.checkbox("Inclure les week-ends dans la prévision", value=True)
        st.caption(f"Durée d'une tranche horaire : {int(slot_duration_sec)} secondes.")

    st.markdown("### Seuils de qualité de service")
    c1, c2, c3 = st.columns(3)
    with c1:
        seuil_attente = st.slider("Attente max acceptable (s)", 5, 120, 20)
    with c2:
        seuil_decroche_pct = st.slider("Taux de décroché max acceptable (%)", 1, 30, 5)
    with c3:
        max_agents = st.slider("Nombre maximum d'agents disponibles", 1, 60, 20)
    seuil_decroche = seuil_decroche_pct / 100.0

    if st.button("🧮 Générer les recommandations", type="primary"):
        slots = sorted(agg["heure_debut"].unique())
        future = build_future_skeleton(
            agg["date"].max(), horizon, slots, country="FR",
            exclude_holidays=True, include_weekends=include_weekends,
        )
        if future.empty:
            st.error("Aucune date future a prevoir avec ces parametres (essayez d'inclure les week-ends).")
        else:
            with st.spinner(f"Entraînement de « {model_name} » sur tout l'historique disponible..."):
                model = MODEL_REGISTRY[model_name]()
                model.fit(agg)
                result = model.predict(future, extra_history=agg[["date", "heure_debut", "volume"]])
                quality = estimate_future_dmt_and_quality(agg, future)

            staffing_rows = []
            for i, row in future.reset_index(drop=True).iterrows():
                q = quality.iloc[i]
                rec = recommend_staffing(
                    volume_pred=max(float(result.pred_mean[i]), 0.0),
                    dmt_moyenne_sec=float(q["dmt_moyenne"]),
                    attente_historique_sec=float(q["attente_moyenne"]),
                    decroche_historique=float(q["taux_decroche"]),
                    slot_duration_sec=slot_duration_sec,
                    seuil_attente_sec=seuil_attente,
                    seuil_decroche=seuil_decroche,
                    max_agents=max_agents,
                )
                staffing_rows.append({
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "heure_debut": row["heure_debut"],
                    "tranche": row["tranche"],
                    "jour_semaine": row["jour_semaine"],
                    "volume_predit": round(float(result.pred_mean[i]), 1),
                    "volume_ic_inf": round(float(result.pred_lower[i]), 1) if result.pred_lower is not None else np.nan,
                    "volume_ic_sup": round(float(result.pred_upper[i]), 1) if result.pred_upper is not None else np.nan,
                    "dmt_moyenne_s": round(float(q["dmt_moyenne"]), 1),
                    "agents_recommandes": rec.agents_recommandes,
                    "attente_estimee_s": rec.attente_estimee_sec,
                    "taux_decroche_estime": rec.taux_decroche_estime,
                    "depasse_max": rec.depasse_max_agents,
                })

            staffing_df = pd.DataFrame(staffing_rows)
            st.session_state["staffing_df"] = staffing_df
            st.session_state["staffing_has_interval"] = bool(
                result.pred_lower is not None and not np.all(np.isnan(result.pred_lower))
            )

    staffing_df = st.session_state.get("staffing_df")
    if staffing_df is not None and not staffing_df.empty:
        n_warn = int(staffing_df["depasse_max"].sum())
        if n_warn > 0:
            st.warning(
                f"⚠️ {n_warn} tranche(s) horaire(s) nécessiteraient plus d'agents que le maximum "
                f"disponible ({max_agents}). La qualité de service ne pourra pas y être garantie."
            )

        display_cols = ["date", "tranche", "jour_semaine", "volume_predit"]
        if st.session_state.get("staffing_has_interval"):
            display_cols += ["volume_ic_inf", "volume_ic_sup"]
        display_cols += ["dmt_moyenne_s", "agents_recommandes", "attente_estimee_s", "taux_decroche_estime"]

        st.subheader("Tableau de recommandation")
        st.dataframe(staffing_df[display_cols], width='stretch', hide_index=True)

        st.subheader("Agents recommandés par date et tranche")
        pivot = staffing_df.pivot_table(index="heure_debut", columns="date", values="agents_recommandes")
        fig = px.imshow(
            pivot, labels=dict(x="Date", y="Heure de début", color="Agents"),
            aspect="auto", color_continuous_scale="Oranges",
        )
        st.plotly_chart(fig, width='stretch')

        csv_bytes = staffing_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Exporter les recommandations (CSV)", csv_bytes, "recommandations_staffing.csv", "text/csv"
        )
    else:
        st.info("Choisissez un modèle et un horizon, puis cliquez sur « Générer les recommandations ».")


# ---------------------------------------------------------------------------
# Page 5 : Donnees synthetiques
# ---------------------------------------------------------------------------

def page_synthetic():
    st.header("🧪 Données synthétiques (démo)")
    st.markdown(
        "Générez un jeu de données réaliste pour tester l'application sans "
        "avoir besoin de données réelles. Le volume d'appels suit un "
        "processus de Poisson dont l'intensité varie par jour de la semaine "
        "et par heure ; le temps d'attente suit une loi exponentielle ; la "
        "DMT varie par agent, avec une amélioration progressive optionnelle."
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début", value=pd.Timestamp("2026-01-01"))
        end_date = st.date_input("Date de fin", value=pd.Timestamp("2026-03-31"))
        n_agents = st.slider("Nombre d'agents", 2, 20, 5)
        base_lambda = st.slider("Volume moyen par heure (référence)", 5, 100, 18)
    with col2:
        mean_wait = st.slider("Temps d'attente moyen visé (s)", 5, 90, 25)
        abandon_threshold = st.slider("Seuil d'abandon de l'appel (s)", 20, 180, 60)
        mean_talk = st.slider("Durée moyenne de communication (s)", 60, 600, 210)
        mean_post_call = st.slider("Durée moyenne de post-appel (s)", 10, 180, 45)

    learning = st.checkbox("Simuler une amélioration progressive de la DMT (apprentissage)", value=True)
    seed = st.number_input("Graine aléatoire (reproductibilité)", value=42, step=1)

    if st.button("🎲 Générer les données", type="primary"):
        if pd.Timestamp(end_date) <= pd.Timestamp(start_date):
            st.error("La date de fin doit être après la date de début.")
        else:
            agents = [f"A{str(i + 1).zfill(3)}" for i in range(n_agents)]
            with st.spinner("Génération en cours..."):
                df = generate_synthetic_calls(
                    str(start_date), str(end_date), agents=agents, base_lambda=base_lambda,
                    mean_wait_sec=mean_wait, wait_threshold_for_abandon_sec=abandon_threshold,
                    mean_talk_sec=mean_talk, mean_post_call_sec=mean_post_call,
                    learning_curve=learning, random_seed=int(seed),
                )
            st.session_state["synthetic_df"] = df

    synthetic_df = st.session_state.get("synthetic_df")
    if synthetic_df is not None and not synthetic_df.empty:
        st.success(
            f"{len(synthetic_df):,} appels générés entre {synthetic_df['date_heure'].min()} "
            f"et {synthetic_df['date_heure'].max()}.".replace(",", " ")
        )
        st.dataframe(synthetic_df.head(50), width='stretch', hide_index=True)

        csv_bytes = synthetic_df.to_csv(index=False).encode("utf-8")
        col_dl, col_use = st.columns(2)
        with col_dl:
            st.download_button("⬇️ Télécharger en CSV", csv_bytes, "donnees_synthetiques.csv", "text/csv")
        with col_use:
            if st.button("✅ Utiliser ces données dans l'application"):
                st.session_state["raw_df"] = synthetic_df.copy()
                st.session_state.pop("backtest_results", None)
                st.session_state.pop("staffing_df", None)
                st.success("Données chargées ! Utilisez le menu de navigation pour explorer les autres pages.")
    else:
        st.info("Ajustez les paramètres ci-dessus puis cliquez sur « Générer les données ».")


# ---------------------------------------------------------------------------
# Point d'entree principal
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Prévision d'appels & Staffing", page_icon="📞", layout="wide")
    st.title("📞 Prévision du volume d'appels & recommandation de staffing")

    st.sidebar.header("⚙️ Configuration")
    uploaded = st.sidebar.file_uploader("Importer un fichier CSV d'appels", type=["csv"])
    if uploaded is not None:
        raw = load_calls_csv(uploaded)
        validation = validate_calls_dataframe(raw)
        if not validation.is_valid:
            st.sidebar.error(f"Colonnes manquantes : {validation.missing_columns}")
        else:
            st.session_state["raw_df"] = raw
            st.session_state.pop("backtest_results", None)
            st.session_state.pop("staffing_df", None)
            st.sidebar.success(f"{validation.n_rows} lignes chargées.")
            for w in validation.warnings:
                st.sidebar.warning(w)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres globaux")
    slot_minutes = st.sidebar.selectbox(
        "Taille des tranches horaires", [60, 30], format_func=lambda m: f"{m} minutes"
    )
    wait_threshold_sec = st.sidebar.slider(
        "Seuil d'attente pour 'décroché tardif' (s)", 5, 60, 20,
        help="Un appel répondu après un temps d'attente supérieur à ce seuil est compté comme 'décroché tardif'.",
    )
    exclude_holidays = st.sidebar.checkbox("Exclure les jours fériés (France)", value=True)

    pages = [
        "Exploration des données",
        "Validation de Poisson",
        "Prédiction et comparaison",
        "Recommandation de staffing",
        "Données synthétiques",
    ]
    page = st.sidebar.radio("📂 Navigation", pages)

    if "raw_df" not in st.session_state:
        st.info(
            "👈 Importez un fichier CSV dans la barre latérale, ou rendez-vous sur la page "
            "**Données synthétiques** pour générer un exemple de démonstration."
        )
        if page == "Données synthétiques":
            page_synthetic()
        return

    raw_df = st.session_state["raw_df"]
    feat, agg, dmt_agent = process_pipeline(raw_df, slot_minutes, wait_threshold_sec, exclude_holidays, "FR")
    slot_duration_sec = slot_minutes * 60

    if page == "Exploration des données":
        page_exploration(feat, agg, dmt_agent)
    elif page == "Validation de Poisson":
        page_poisson(agg)
    elif page == "Prédiction et comparaison":
        page_prediction(agg)
    elif page == "Recommandation de staffing":
        page_staffing(agg, slot_duration_sec)
    elif page == "Données synthétiques":
        page_synthetic()


if __name__ == "__main__":
    main()
