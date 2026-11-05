"""
app.py
======
Application Streamlit : prévision du volume d'appels entrants et
recommandation du nombre d'agents nécessaires (staffing), par date et
tranche horaire.

Lancement :
    streamlit run app.py

Structure (voir README.md pour le détail) :
    - Exploration des données
    - Validation de la distribution de Poisson
    - Prévision et comparaison de modèles
    - Recommandation de staffing
    - Données synthetiques (demo)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate_synthetic_calls
from models import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    SLOW_MODELS,
    noise_limits,
    walk_forward_evaluate,
)
from staffing import estimate_future_dmt, recommend_staffing, simulate_queue_kpis
from utils import (
    JOURS_SEMAINE_FR,
    STATUT_REPONDU,
    add_time_features,
    aggregate_by_slot,
    build_future_skeleton,
    dispersion_test,
    dmt_by_agent,
    filter_by_agent_and_date,
    filter_holidays,
    hour_label,
    kpi_summary,
    load_calls_csv,
    poisson_pmf_curve,
    poisson_validation_report,
    pooled_dispersion_test,
    validate_calls_dataframe,
)

EXAMPLE_FILE = Path(__file__).parent / "exemple_donnees_appels.csv"


# ---------------------------------------------------------------------------
# Pipeline de prétraitement (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Prétraitement des données...")
def process_pipeline(raw_df: pd.DataFrame, slot_minutes: int, wait_threshold_sec: float,
                      exclude_holidays: bool, country: str = "FR"):
    feat = add_time_features(raw_df, slot_minutes=slot_minutes)
    feat = filter_holidays(feat, country=country, exclude=exclude_holidays)
    agg = aggregate_by_slot(feat, wait_threshold_sec=wait_threshold_sec, slot_minutes=slot_minutes)
    dmt_agent = dmt_by_agent(feat)
    return feat, agg, dmt_agent


def reset_results():
    for key in ("backtest_results", "staffing_df", "staffing_meta"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Page 1 : Exploration des données
# ---------------------------------------------------------------------------

def page_exploration(feat: pd.DataFrame, agg: pd.DataFrame, dmt_agent: pd.DataFrame):
    st.header("🔍 Exploration des données")

    if agg.empty:
        st.warning("Aucune donnée disponible après filtrage. Vérifiez votre fichier ou vos paramètres.")
        return

    kpis = kpi_summary(agg)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Appels reçus", f"{kpis['volume']:,}".replace(",", " "))
    col2.metric("Niveau de service", f"{kpis['niveau_service'] * 100:.1f} %",
                help="Part des appels reçus qui sont pris dans le délai choisi dans la barre latérale.")
    col3.metric("Taux d'abandon", f"{kpis['taux_abandon'] * 100:.1f} %",
                help="Part des appels reçus où l'appelant raccroche avant d'être pris.")
    col4.metric("DMT moyenne", f"{kpis['dmt_moyenne']:.0f} s",
                help="Durée moyenne de traitement : communication + post-appel, hors attente.")
    col5.metric("Attente moyenne", f"{kpis['attente_moyenne']:.1f} s",
                help="Attente moyenne des appels répondus.")

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
        st.info("Aucun agent identifié dans les données.")

    left, right = st.columns(2)
    with left:
        st.subheader("Temps d'attente moyen par tranche")
        by_slot = agg.groupby("heure_debut").apply(
            lambda g: (g["attente_moyenne"] * g["n_repondu"]).sum() / max(g["n_repondu"].sum(), 1),
            include_groups=False,
        ).rename("attente_moyenne").reset_index()
        by_slot["tranche"] = by_slot["heure_debut"].map(hour_label)
        fig3 = px.bar(
            by_slot, x="tranche", y="attente_moyenne",
            labels={"tranche": "Heure de début", "attente_moyenne": "Attente moyenne (s)"},
        )
        st.plotly_chart(fig3, width='stretch')
    with right:
        st.subheader("Niveau de service et abandon par jour")
        totals = agg.groupby("jour_semaine")[["volume", "n_dans_delai", "n_abandonne"]].sum()
        totals = totals.reindex(JOURS_SEMAINE_FR).dropna()
        by_day = pd.DataFrame({
            "Niveau de service": totals["n_dans_delai"] / totals["volume"],
            "Taux d'abandon": totals["n_abandonne"] / totals["volume"],
        }).reset_index().melt("jour_semaine", var_name="Indicateur", value_name="Part des appels")
        fig4 = px.bar(
            by_day, x="jour_semaine", y="Part des appels", color="Indicateur", barmode="group",
            labels={"jour_semaine": "Jour"},
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
- **Test de dispersion :** vérifie si la variance dépasse la moyenne. Sur une
  seule tranche, avec une douzaine d'observations, ce test détecte peu de
  choses. Le **test global** regroupe toutes les tranches et sert de décision.
            """
        )

    if agg.empty:
        st.warning("Aucune donnée disponible après filtrage.")
        return

    pooled = pooled_dispersion_test(agg)
    st.subheader("Test global")
    if not pooled["valide"]:
        st.info(pooled["message"])
    elif pooled["p_value"] >= 0.05:
        st.success(pooled["message"])
    else:
        st.warning(pooled["message"])

    st.subheader("Détail par tranche")
    jours_disponibles = [j for j in JOURS_SEMAINE_FR if j in agg["jour_semaine"].unique()]
    if not jours_disponibles:
        st.warning("Aucun jour disponible dans les données.")
        return

    jour = st.selectbox("Jour de la semaine", jours_disponibles)
    heures = sorted(agg.loc[agg["jour_semaine"] == jour, "heure_debut"].unique())
    heure = st.selectbox("Tranche horaire à visualiser", heures, format_func=hour_label)

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
            title=f"{jour}, tranche de {hour_label(heure)}",
        )
        st.plotly_chart(fig, width='stretch')

        test_result = dispersion_test(obs.values)
        st.info(test_result["message"])

    st.markdown("### Tableau récapitulatif — tous les jours et tranches horaires")
    st.dataframe(poisson_validation_report(agg), width='stretch', hide_index=True)
    st.caption(
        "`p_ajustee` corrige les comparaisons multiples : à interpréter à la place de `p_value`. "
        "Les colonnes sont vides quand une tranche compte trop peu d'observations."
    )


# ---------------------------------------------------------------------------
# Page 3 : Prévision et comparaison de modèles
# ---------------------------------------------------------------------------

def page_prediction(agg: pd.DataFrame):
    st.header("📈 Prévision et comparaison de modèles")

    if agg.empty or agg["date"].nunique() < 35:
        st.warning("Historique court (moins de 5 semaines) : les scores seront peu fiables.")

    with st.expander("ℹ️ Quels modèles, et pourquoi ?"):
        for name, cls in MODEL_REGISTRY.items():
            st.markdown(f"**{name}**")
            st.caption(cls.description)

    model_names = list(MODEL_REGISTRY.keys())
    default_models = [name for name in model_names if name not in SLOW_MODELS]
    selected = st.multiselect("Modèles à comparer", model_names, default=default_models)
    col1, col2 = st.columns(2)
    test_days = col1.slider("Jours testés par repli", 7, 28, 14)
    n_folds = col2.slider("Nombre de replis (validation glissante)", 1, 5, 3)

    if st.button("🚀 Lancer la comparaison", type="primary") and selected:
        results = {}
        progress = st.progress(0.0)
        for i, name in enumerate(selected):
            try:
                with st.spinner(f"Validation glissante : {name}..."):
                    results[name] = walk_forward_evaluate(MODEL_REGISTRY[name], agg, test_days, n_folds)
            except Exception as exc:
                st.warning(f"{name} : échec ({exc})")
            progress.progress((i + 1) / len(selected))
        progress.empty()
        st.session_state["backtest_results"] = results

    results = st.session_state.get("backtest_results")
    if not results:
        st.info("Sélectionnez des modèles puis cliquez sur « Lancer la comparaison ».")
        return

    metrics_df = pd.DataFrame({name: res["metrics"] for name, res in results.items()}).T.sort_values("RMSE")
    best_model = metrics_df["RMSE"].idxmin()
    best = results[best_model]
    limit = noise_limits(best["test_df"]["volume"], best["predictions"]["pred_mean"])

    table = pd.concat([pd.DataFrame([limit], index=["Limite théorique (hasard)"]), metrics_df])
    st.subheader("Comparaison des métriques")
    st.dataframe(table.style.format("{:.2f}", na_rep="—"), width='stretch')
    st.success(
        f"Meilleur modèle : **{best_model}**. Son RMSE ({metrics_df.loc[best_model, 'RMSE']:.2f}) "
        f"est à comparer à la limite théorique ({limit['RMSE']:.2f}) : même un modèle connaissant "
        "le vrai rythme d'appels ne ferait pas mieux, car le reste est du hasard."
    )
    for name, res in results.items():
        rhat = res["diagnostics"].get("rhat_max")
        if rhat is not None and (rhat > 1.01 or res["diagnostics"]["divergences"] > 0):
            st.warning(
                f"{name} : convergence à vérifier (R-hat max = {rhat:.3f}, "
                f"{res['diagnostics']['divergences']} divergences)."
            )

    with st.expander("ℹ️ Comment interpréter ces métriques ?"):
        st.markdown(
            """
- **RMSE / MAE** : erreur moyenne par tranche, en nombre d'appels.
- **WAPE** : erreur totale divisée par le volume total. Une tranche compte peu
  d'appels, donc ce pourcentage reste élevé même pour une excellente prévision.
- **WAPE jour** : la même erreur sur le total de chaque journée, plus lisible
  pour planifier les équipes.
- **R²** : part de la variation expliquée. Il ne peut pas dépasser la limite théorique.
- **Couverture IC95%** : part des volumes réels dans l'intervalle à 95 % (cible : 95 %).
- **Largeur moyenne IC** : à couverture égale, un intervalle étroit est plus informatif.
            """
        )

    st.subheader("Prévisions vs données réelles")
    model_choices = list(results.keys())
    chosen = st.selectbox("Modèle à visualiser", model_choices, index=model_choices.index(best_model))
    plot_df = results[chosen]["test_df"].join(results[chosen]["predictions"])
    by_date = plot_df.groupby("date")[["volume", "pred_mean"]].sum().reset_index()

    fig = go.Figure()
    fig.add_scatter(x=by_date["date"], y=by_date["volume"], mode="lines+markers", name="Réel")
    fig.add_scatter(x=by_date["date"], y=by_date["pred_mean"], mode="lines+markers", name="Prévu")
    fig.update_layout(xaxis_title="Date", yaxis_title="Appels par jour")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(
        plot_df[["date", "tranche", "volume", "pred_mean", "pred_lower", "pred_upper"]],
        width='stretch', hide_index=True,
    )


# ---------------------------------------------------------------------------
# Page 4 : Recommandation de staffing
# ---------------------------------------------------------------------------

def service_time_cv(feat: pd.DataFrame) -> float:
    """Coefficient de variation de la DMT, repris dans la simulation de contrôle."""
    answered = feat[feat["statut"] == STATUT_REPONDU]
    durations = answered["post_appel"].fillna(0) + answered["temps_communication"].fillna(0)
    return float(durations.std() / durations.mean()) if len(durations) > 1 else 0.5


def page_staffing(agg: pd.DataFrame, feat: pd.DataFrame, slot_minutes: int, delai_sec: float):
    st.header("👥 Recommandation de staffing")

    if agg.empty:
        st.warning("Aucune donnée disponible après filtrage.")
        return

    slot_duration_sec = slot_minutes * 60
    with st.expander("ℹ️ Comment est calculée la recommandation ?"):
        st.markdown(
            f"""
1. **Charge** : volume prévu × DMT ÷ durée de la tranche, en erlangs.
2. **Erlang C** : pour chaque effectif, estime la probabilité d'attendre et la part
   des appels pris en moins de {delai_sec:.0f} s.
3. **Recommandation** : le plus petit effectif qui respecte le niveau de service et
   l'occupation maximale.
4. **Contrôle** : une simulation de la tranche la plus chargée vérifie le résultat.

*Erlang C ignore les abandons et suppose une file illimitée. Le délai de réponse se
règle dans la barre latérale.*
            """
        )

    default_index = list(MODEL_REGISTRY.keys()).index(DEFAULT_MODEL)
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox("Modèle de prévision", list(MODEL_REGISTRY.keys()), index=default_index)
        horizon = st.slider("Horizon de prévision (jours)", 1, 30, 7)
    with col2:
        basis = st.radio(
            "Volume à couvrir", ["Volume prévu", "Borne haute de l'intervalle à 95 %"],
            help="La borne haute protège contre les pics, mais demande plus d'agents.",
        )
        include_weekends = st.checkbox("Inclure les week-ends", value=True)

    st.markdown("### Objectifs")
    c1, c2, c3 = st.columns(3)
    objectif_pct = c1.slider(f"Appels pris en moins de {delai_sec:.0f} s (%)", 50, 99, 80)
    occupation_pct = c2.slider("Occupation maximale des agents (%)", 50, 100, 85)
    max_agents = c3.slider("Agents disponibles au maximum", 1, 60, 20)

    if st.button("🧮 Générer les recommandations", type="primary"):
        future = build_future_skeleton(
            agg["date"].max(), horizon, sorted(agg["heure_debut"].unique()), country="FR",
            exclude_holidays=True, include_weekends=include_weekends, slot_minutes=slot_minutes,
        )
        if future.empty:
            st.error("Aucune date à prévoir avec ces paramètres : essayez d'inclure les week-ends.")
        else:
            with st.spinner(f"Entraînement de « {model_name} »..."):
                forecast = MODEL_REGISTRY[model_name]().fit(agg).predict(future)
                dmt = estimate_future_dmt(agg, future)

            target = forecast.pred_mean if basis == "Volume prévu" else forecast.pred_upper
            rows = []
            for i, row in future.reset_index(drop=True).iterrows():
                rec = recommend_staffing(
                    volume_pred=float(target[i]),
                    dmt_sec=float(dmt[i]),
                    slot_duration_sec=slot_duration_sec,
                    objectif_service=objectif_pct / 100,
                    delai_reponse_sec=delai_sec,
                    occupation_max=occupation_pct / 100,
                    max_agents=max_agents,
                )
                rows.append({
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "heure_debut": row["heure_debut"],
                    "tranche": row["tranche"],
                    "jour_semaine": row["jour_semaine"],
                    "volume_prevu": round(float(forecast.pred_mean[i]), 1),
                    "volume_ic_inf": round(float(forecast.pred_lower[i]), 1),
                    "volume_ic_sup": round(float(forecast.pred_upper[i]), 1),
                    "volume_couvert": round(float(target[i]), 1),
                    "dmt_s": round(float(dmt[i]), 1),
                    "charge_erlangs": rec.charge_erlangs,
                    "agents_recommandes": rec.agents_recommandes,
                    "niveau_service_estime": rec.niveau_service_estime,
                    "attente_estimee_s": rec.attente_estimee_sec,
                    "occupation_estimee": rec.occupation_estimee,
                    "depasse_max": rec.depasse_max_agents,
                })
            st.session_state["staffing_df"] = pd.DataFrame(rows)
            st.session_state["staffing_meta"] = {"max_agents": max_agents}

    staffing_df = st.session_state.get("staffing_df")
    if staffing_df is None or staffing_df.empty:
        st.info("Choisissez un modèle et un horizon, puis cliquez sur « Générer les recommandations ».")
        return

    n_warn = int(staffing_df["depasse_max"].sum())
    if n_warn:
        st.warning(
            f"⚠️ {n_warn} tranche(s) demanderaient plus de "
            f"{st.session_state['staffing_meta']['max_agents']} agents : "
            "l'objectif de service n'y sera pas atteint."
        )

    hours_per_slot = slot_minutes / 60
    c1, c2, c3 = st.columns(3)
    c1.metric("Pic d'effectif", f"{int(staffing_df['agents_recommandes'].max())} agents")
    c2.metric("Heures-agent sur la période", f"{staffing_df['agents_recommandes'].sum() * hours_per_slot:.0f} h")
    c3.metric("Occupation moyenne", f"{staffing_df['occupation_estimee'].mean() * 100:.0f} %")

    st.subheader("Agents recommandés par date et tranche")
    pivot = staffing_df.pivot_table(index="heure_debut", columns="date", values="agents_recommandes")
    pivot.index = [hour_label(h) for h in pivot.index]
    fig = px.imshow(
        pivot, labels=dict(x="Date", y="Heure de début", color="Agents"),
        aspect="auto", color_continuous_scale="Oranges", text_auto=True,
    )
    fig.update_xaxes(type="category")  # un jour férié retiré ne doit pas laisser de vide
    st.plotly_chart(fig, width='stretch')

    peak = staffing_df.loc[staffing_df["charge_erlangs"].idxmax()]
    simulated = simulate_queue_kpis(
        peak["volume_couvert"], peak["dmt_s"], int(peak["agents_recommandes"]),
        slot_duration_sec=slot_duration_sec, delai_reponse_sec=delai_sec,
        cv_service=service_time_cv(feat),
    )
    st.subheader("Contrôle par simulation de la tranche la plus chargée")
    st.caption(f"{peak['tranche']} : {peak['volume_couvert']:.1f} appels, {int(peak['agents_recommandes'])} agents.")
    c1, c2 = st.columns(2)
    c1.metric(
        "Niveau de service simulé", f"{simulated['niveau_service'] * 100:.1f} %",
        f"Erlang C : {peak['niveau_service_estime'] * 100:.1f} %", delta_color="off", delta_arrow="off",
    )
    c2.metric(
        "Attente moyenne simulée", f"{simulated['attente_moyenne_sec']:.1f} s",
        f"Erlang C : {peak['attente_estimee_s']:.1f} s", delta_color="off", delta_arrow="off",
    )

    st.subheader("Tableau de recommandation")
    columns = [column for column in staffing_df.columns if column not in ("heure_debut", "depasse_max")]
    st.dataframe(staffing_df[columns], width='stretch', hide_index=True)
    st.download_button(
        "⬇️ Exporter les recommandations (CSV)", staffing_df[columns].to_csv(index=False).encode("utf-8"),
        "recommandations_staffing.csv", "text/csv",
    )


# ---------------------------------------------------------------------------
# Page 5 : Données synthetiques
# ---------------------------------------------------------------------------

def page_synthetic():
    st.header("🧪 Données synthétiques (démo)")
    st.markdown(
        "Générez un jeu de données pour tester l'application sans données "
        "réelles. Le volume suit un processus de Poisson qui varie selon le "
        "jour et l'heure, et les appels traversent une file d'attente simulée : "
        "l'attente et les abandons dépendent donc de la charge et du planning."
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début", value=pd.Timestamp("2026-01-01"))
        end_date = st.date_input("Date de fin", value=pd.Timestamp("2026-03-31"))
        n_agents = st.slider("Nombre d'agents", 2, 20, 5)
        base_lambda = st.slider("Volume moyen par heure (référence)", 5, 100, 18)
    with col2:
        mean_patience = st.slider("Patience moyenne des appelants (s)", 30, 300, 150)
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
                    mean_patience_sec=mean_patience,
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
                reset_results()
                st.success("Données chargées ! Utilisez le menu de navigation pour explorer les autres pages.")
    else:
        st.info("Ajustez les paramètres ci-dessus puis cliquez sur « Générer les données ».")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Prévision d'appels & Staffing", page_icon="📞", layout="wide")
    st.title("📞 Prévision du volume d'appels & recommandation de staffing")

    st.sidebar.header("⚙️ Configuration")
    uploaded = st.sidebar.file_uploader("Importer un fichier CSV d'appels", type=["csv"])
    if uploaded is not None and st.session_state.get("uploaded_id") != uploaded.file_id:
        raw = load_calls_csv(uploaded)
        validation = validate_calls_dataframe(raw)
        if not validation.is_valid:
            st.sidebar.error(f"Colonnes manquantes : {validation.missing_columns}")
        else:
            st.session_state["raw_df"] = raw
            st.session_state["uploaded_id"] = uploaded.file_id
            reset_results()
            st.sidebar.success(f"{validation.n_rows} lignes chargées.")
            for warning in validation.warnings:
                st.sidebar.warning(warning)

    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = load_calls_csv(EXAMPLE_FILE)
        st.session_state["demo_data"] = True
    if st.session_state.get("uploaded_id"):
        st.session_state["demo_data"] = False
    if st.session_state.get("demo_data"):
        st.sidebar.info("Données d'exemple synthétiques chargées.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres globaux")
    slot_minutes = st.sidebar.selectbox(
        "Taille des tranches horaires", [60, 30], format_func=lambda m: f"{m} minutes"
    )
    wait_threshold_sec = st.sidebar.slider(
        "Délai de réponse visé (s)", 5, 60, 20,
        help="Le niveau de service est la part des appels pris dans ce délai.",
    )
    exclude_holidays = st.sidebar.checkbox("Exclure les jours fériés (France)", value=True)

    pages = [
        "Exploration des données",
        "Validation de Poisson",
        "Prévision et comparaison",
        "Recommandation de staffing",
        "Données synthétiques",
    ]
    page = st.sidebar.radio("📂 Navigation", pages)

    raw_df = st.session_state["raw_df"]
    feat, agg, dmt_agent = process_pipeline(raw_df, slot_minutes, wait_threshold_sec, exclude_holidays, "FR")

    if page == "Exploration des données":
        page_exploration(feat, agg, dmt_agent)
    elif page == "Validation de Poisson":
        page_poisson(agg)
    elif page == "Prévision et comparaison":
        page_prediction(agg)
    elif page == "Recommandation de staffing":
        page_staffing(agg, feat, slot_minutes, wait_threshold_sec)
    elif page == "Données synthétiques":
        page_synthetic()


if __name__ == "__main__":
    main()
