import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

warnings.filterwarnings("ignore")

from sklearn.inspection import permutation_importance

# ── Configuration page ───────────────────────────────────────
st.set_page_config(
    page_title="Marketing ROI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16192a 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label {
        font-size: 12px;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #f9fafb;
    }
    .metric-card .delta {
        font-size: 12px;
        color: #10b981;
        margin-top: 4px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #f9fafb;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2563eb;
        display: inline-block;
    }

    .predict-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
        border: 1px solid #2563eb;
        border-radius: 16px;
        padding: 28px;
        margin-top: 16px;
    }
    .predict-result {
        font-size: 42px;
        font-weight: 700;
        color: #60a5fa;
        text-align: center;
    }
    .predict-label {
        font-size: 13px;
        color: #93c5fd;
        text-align: center;
        margin-top: 4px;
    }

    .roi-box {
        background: linear-gradient(135deg, #1e4a3a 0%, #1a3a2a 100%);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 28px;
        margin-top: 16px;
    }
    .roi-result {
        font-size: 42px;
        font-weight: 700;
        color: #34d399;
        text-align: center;
    }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-blue  { background: #1d4ed8; color: #bfdbfe; }
    .badge-green { background: #065f46; color: #a7f3d0; }
    .badge-red   { background: #7f1d1d; color: #fecaca; }

    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1f2937;
    }
    div[data-testid="stSidebar"] .stRadio label {
        color: #d1d5db !important;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes ───────────────────────────────────────────────
INFLUENCER_MAPPING = {"Nano": 0, "Micro": 1, "Macro": 2, "Mega": 3}
INFLUENCER_ORDER   = ["Nano", "Micro", "Macro", "Mega"]
FEATURES           = ["TV", "Radio", "Social Media", "Influencer_encoded"]
COLORS             = ["#3b82f6", "#10b981", "#ef4444", "#8b5cf6"]
MODEL_NAMES        = ["Régression Linéaire", "Random Forest", "Gradient Boosting", "MLP (Deep Learning)"]


DATA_PATH          = os.path.join(PROJECT_ROOT, "data", "processed", "marketing_and_sales_clean.csv")
MODEL_SALES_PATH   = os.path.join(PROJECT_ROOT, "model", "best_model.pkl")
MODEL_ROI_PATH     = os.path.join(PROJECT_ROOT, "model", "best_model_roi.pkl")
PREPROC_SALES_PATH = os.path.join(PROJECT_ROOT, "model", "preprocessor.pkl")
PREPROC_ROI_PATH   = os.path.join(PROJECT_ROOT, "model", "preprocessor_roi.pkl")
# ── Chargement données & modèles ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Influencer_encoded"] = df["Influencer"].map(INFLUENCER_MAPPING)
    if "ROI" not in df.columns:
        df["Total_Budget"] = df["TV"] + df["Radio"] + df["Social Media"]
        df["ROI"] = (df["Sales"] / df["Total_Budget"]).round(6)
    return df

@st.cache_resource
def load_models():
    models = {}
    for key, path in [("sales", MODEL_SALES_PATH), ("roi", MODEL_ROI_PATH)]:
        if os.path.exists(path):
            models[key] = joblib.load(path)
    for key, path in [("preproc_sales", PREPROC_SALES_PATH), ("preproc_roi", PREPROC_ROI_PATH)]:
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

# Résultats des modèles (tes vraies métriques)
RESULTS_SALES = pd.DataFrame([
    {"Modèle": "Régression Linéaire",  "MAE": 2312.78,  "RMSE": 2884.92,  "R²": 0.9990, "CV R²": 0.9990, "CV std": 0.0000},
    {"Modèle": "Gradient Boosting",    "MAE": 2355.06,  "RMSE": 2951.50,  "R²": 0.9990, "CV R²": 0.9989, "CV std": 0.0000},
    {"Modèle": "Random Forest",        "MAE": 2729.71,  "RMSE": 3357.43,  "R²": 0.9987, "CV R²": 0.9987, "CV std": 0.0000},
    {"Modèle": "MLP (Deep Learning)",  "MAE": 4661.89,  "RMSE": 6282.40,  "R²": 0.9953, "CV R²": 0.9281, "CV std": 0.0034},
])

RESULTS_ROI = pd.DataFrame([
    {"Modèle": "MLP (Deep Learning)",  "MAE": 0.046444, "RMSE": 0.069437, "R²": 0.9385, "CV R²": 0.9329, "CV std": 0.0050},
    {"Modèle": "Random Forest",        "MAE": 0.049626, "RMSE": 0.072635, "R²": 0.9327, "CV R²": 0.9185, "CV std": 0.0078},
    {"Modèle": "Gradient Boosting",    "MAE": 0.050853, "RMSE": 0.073724, "R²": 0.9306, "CV R²": 0.9217, "CV std": 0.0084},
    {"Modèle": "Régression Linéaire",  "MAE": 0.102991, "RMSE": 0.151428, "R²": 0.7074, "CV R²": 0.7093, "CV std": 0.0131},
])

# ── Sidebar navigation ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Marketing ROI")
    st.markdown("*Système intelligent d'optimisation budgétaire*")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Accueil",
        "🎯 Prédiction",
        "📈 Comparaison des modèles",
        "🔍 Interprétabilité",
    ])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("TV · Radio · Social Media · Influencer → Sales / ROI")
    st.markdown("---")
    st.markdown("*EFREI M2 — Data Science 2025-26*")

df      = load_data()
models  = load_models()

# ════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown("# 📊 Marketing ROI Dashboard")
    st.markdown("Plateforme intelligente d'optimisation du retour sur investissement marketing multicanal.")
    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Campagnes analysées</div>
            <div class="value">{len(df)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Ventes moyennes</div>
            <div class="value">{df['Sales'].mean():,.0f}</div>
            <div class="delta">en milliers</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">ROI moyen</div>
            <div class="value">{df['ROI'].mean():.2f}x</div>
            <div class="delta">retour sur budget</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        best_influencer = df.groupby("Influencer")["ROI"].mean().idxmax()
        st.markdown(f"""<div class="metric-card">
            <div class="label">Meilleur influenceur</div>
            <div class="value">{best_influencer}</div>
            <div class="delta">ROI le plus élevé</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Graphiques EDA
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Distribution des ventes</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d2e")
        ax.hist(df["Sales"], bins=30, color="#3b82f6", edgecolor="#0f1117", alpha=0.85)
        ax.set_xlabel("Ventes (milliers)", color="#9ca3af")
        ax.set_ylabel("Fréquence", color="#9ca3af")
        ax.tick_params(colors="#6b7280")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">ROI par type d\'influenceur</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d2e")
        roi_by_inf = df.groupby("Influencer")["ROI"].mean().sort_values(ascending=False)
        bars = ax.bar(roi_by_inf.index, roi_by_inf.values, color=COLORS, alpha=0.85)
        ax.set_xlabel("Type d'influenceur", color="#9ca3af")
        ax.set_ylabel("ROI moyen", color="#9ca3af")
        ax.tick_params(colors="#6b7280")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")
        for bar, val in zip(bars, roi_by_inf.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", color="#f9fafb", fontsize=10)
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Budget TV vs Ventes</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d2e")
        ax.scatter(df["TV"], df["Sales"], alpha=0.4, color="#3b82f6", s=15)
        ax.set_xlabel("Budget TV (milliers)", color="#9ca3af")
        ax.set_ylabel("Ventes (milliers)", color="#9ca3af")
        ax.tick_params(colors="#6b7280")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Budget Total vs ROI</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d2e")
        ax.scatter(df["Total_Budget"], df["ROI"], alpha=0.4, color="#10b981", s=15)
        ax.set_xlabel("Budget Total (milliers)", color="#9ca3af")
        ax.set_ylabel("ROI", color="#9ca3af")
        ax.tick_params(colors="#6b7280")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")
        st.pyplot(fig)
        plt.close()

    # Aperçu données
    st.markdown('<div class="section-title">Aperçu des données</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["TV", "Radio", "Social Media", "Influencer", "Sales", "ROI"]].head(10),
        use_container_width=True
    )

# ════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION
# ════════════════════════════════════════════════════════════
elif page == "🎯 Prédiction":
    st.markdown("# 🎯 Simulateur Budgétaire")
    st.markdown("Saisissez une combinaison de budgets marketing pour obtenir une prédiction des ventes et du ROI.")
    st.markdown("---")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### ⚙️ Paramètres budgétaires")

        tv           = st.slider("📺 Budget TV (millions)",          0.0, 10.0, 2.0, 0.1)
        radio        = st.slider("📻 Budget Radio (millions)",       0.0, 5.0,  1.0, 0.1)
        social_media = st.slider("📱 Budget Social Media (millions)", 0.0, 5.0, 0.5, 0.1)
        influencer   = st.selectbox("🌟 Type d'Influenceur", INFLUENCER_ORDER)

        total_budget = tv + radio + social_media
        st.markdown(f"**Budget Total : {total_budget:.1f} M€**")

        predict_btn = st.button("🚀 Lancer la prédiction", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Résultats")

        if predict_btn:
            influencer_enc = INFLUENCER_MAPPING[influencer]
            input_df = pd.DataFrame([{
                "TV"          : tv * 1000,
                "Radio"       : radio * 1000,
                "Social Media": social_media * 1000,
                "Influencer_encoded": influencer_enc,
            }])

            # Prédiction Sales
            if "sales" in models and "preproc_sales" in models:
                input_proc   = models["preproc_sales"].transform(input_df)
                pred_sales   = models["sales"].predict(input_proc)[0]
                sales_type   = type(models["sales"]).__name__
            else:
                pred_sales = None
                sales_type = "N/A"

            # Prédiction ROI
            if "roi" in models and "preproc_roi" in models:
                input_proc_roi = models["preproc_roi"].transform(input_df)
                pred_roi       = models["roi"].predict(input_proc_roi)[0]
                roi_type       = type(models["roi"]).__name__
            else:
                pred_roi = None
                roi_type = "N/A"

            if pred_sales is not None:
                st.markdown(f"""<div class="predict-box">
                    <div class="predict-result">{pred_sales/1000:,.2f} M€</div>
<div class="predict-label">Ventes prédites • modèle : {sales_type}</div>
                    <div class="predict-label">Ventes prédites • modèle : {sales_type}</div>
                </div>""", unsafe_allow_html=True)

            rentabilite = (pred_roi - 1) * 100

            st.markdown(f"""<div class="roi-box">
                <div class="roi-result">+{rentabilite:.0f}%</div>
                <div class="predict-label" style="color:#6ee7b7">
                    Rentabilité estimée • modèle : {roi_type}
                </div>
            </div>""", unsafe_allow_html=True)

            # Répartition budgétaire
            st.markdown("#### Répartition du budget")
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#0f1117")
            budgets = {"TV": tv, "Radio": radio, "Social Media": social_media}
            wedges, texts, autotexts = ax.pie(
                budgets.values(),
                labels=budgets.keys(),
                autopct="%1.1f%%",
                colors=["#3b82f6", "#10b981", "#f59e0b"],
                startangle=90,
                textprops={"color": "#f9fafb", "fontsize": 11}
            )
            for at in autotexts:
                at.set_color("#f9fafb")
            st.pyplot(fig)
            plt.close()

        else:
            st.info(" Ajustez les paramètres et cliquez sur **Lancer la prédiction**")

# ════════════════════════════════════════════════════════════
# PAGE 3 — COMPARAISON DES MODÈLES
# ════════════════════════════════════════════════════════════
elif page == "📈 Comparaison des modèles":
    st.markdown("# 📈 Comparaison des modèles")
    st.markdown("---")

    tache = st.radio("Choisir la tâche", ["🛒 Prédiction des Ventes (Sales)", "💰 Prédiction du ROI"], horizontal=True)

    if tache == "🛒 Prédiction des Ventes (Sales)":
        results = RESULTS_SALES
        best_model_name = "Régression Linéaire"
        st.success("✅ **Modèle retenu : Régression Linéaire** — Relation linéaire → modèle simple suffit")
    else:
        results = RESULTS_ROI
        best_model_name = "MLP (Deep Learning)"
        st.success("✅ **Modèle retenu : MLP (Deep Learning)** — Relation non linéaire → DL capture la courbe")

    # Tableau
    st.markdown("### Tableau comparatif")
    styled = results.style\
        .highlight_max(subset=["R²", "CV R²"], color="#065f46")\
        .highlight_min(subset=["MAE", "RMSE", "CV std"], color="#065f46")\
        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}", "CV R²": "{:.4f}", "CV std": "{:.4f}"})
    st.dataframe(styled, use_container_width=True)

    # Graphiques
    st.markdown("### Visualisation des métriques")
    col1, col2, col3 = st.columns(3)

    for col, metric in zip([col1, col2, col3], ["R²", "RMSE", "MAE"]):
        with col:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#1a1d2e")
            colors_bar = ["#10b981" if m == best_model_name else "#3b82f6"
                          for m in results["Modèle"]]
            bars = ax.bar(results["Modèle"], results[metric], color=colors_bar, alpha=0.85)
            ax.set_title(metric, color="#f9fafb", fontweight="bold")
            ax.set_xticklabels(results["Modèle"], rotation=20, ha="right", color="#9ca3af", fontsize=8)
            ax.tick_params(colors="#6b7280")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")
            for bar, val in zip(bars, results[metric]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        f"{val:.4f}", ha="center", color="#f9fafb", fontsize=8)
            st.pyplot(fig)
            plt.close()

    # Analyse
    st.markdown("### 💡 Analyse comparative")
    if tache == "🛒 Prédiction des Ventes (Sales)":
        st.markdown("""
        - **Régression Linéaire** : meilleure sur tous les critères car la relation budgets→ventes est **linéaire par construction**
        - **Gradient Boosting** : quasi ex-aequo mais plus complexe pour aucun gain
        - **Random Forest** : légèrement moins bon, cherche des non-linéarités qui n'existent pas
        - **MLP** : sur-apprend, CV R² chute à 0.9281 → instable sur petit dataset
        """)
    else:
        st.markdown("""
        - **MLP** : meilleur sur tous les critères car la relation ROI→Budget est **non linéaire et décroissante**
        - **Random Forest** : très proche, capture bien la non-linéarité
        - **Gradient Boosting** : équivalent à Random Forest
        - **Régression Linéaire** : R²=0.70 seulement, ne peut pas capturer la courbe
        """)

    # Conclusion clé
    st.markdown("### 🎓 Conclusion pédagogique")
    st.info("""
    **Sur Sales** → Régression Linéaire gagne : relation simple = modèle simple suffit

    **Sur ROI** → MLP gagne : relation complexe non linéaire = besoin d'un modèle puissant

    ➡️ *Le meilleur modèle dépend toujours de la nature des données, pas de sa complexité.*
    """)

# ════════════════════════════════════════════════════════════
# PAGE 4 — INTERPRÉTABILITÉ
# ════════════════════════════════════════════════════════════
elif page == "🔍 Interprétabilité":
    st.markdown("# 🔍 Interprétabilité des modèles")
    st.markdown("Comprendre **pourquoi** le modèle fait ses prédictions.")
    st.markdown("---")

    tache = st.radio("Choisir la tâche", ["🛒 Sales", "💰 ROI"], horizontal=True)

    if tache == "🛒 Sales":
        model_key   = "sales"
        preproc_key = "preproc_sales"
        target      = "Sales"
        best_name   = "Régression Linéaire"
    else:
        model_key   = "roi"
        preproc_key = "preproc_roi"
        target      = "ROI"
        best_name   = "MLP (Deep Learning)"

    st.markdown(f"**Modèle analysé : {best_name}** pour la tâche **{target}**")

    if model_key in models and preproc_key in models:
        model     = models[model_key]
        preproc   = models[preproc_key]
        feat_names = ["TV", "Radio", "Social Media", "Influencer_encoded"]

        # Préparer X
        X = df[["TV", "Radio", "Social Media", "Influencer_encoded"]]
        X_proc = preproc.transform(X)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Permutation Importance")
            try:
                perm = permutation_importance(model, X_proc, df[target],
                                              n_repeats=5, random_state=42)
                df_perm = pd.DataFrame({
                    "Feature"   : feat_names,
                    "Importance": perm.importances_mean,
                }).sort_values("Importance", ascending=True)

                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor("#0f1117")
                ax.set_facecolor("#1a1d2e")
                ax.barh(df_perm["Feature"], df_perm["Importance"],
                        color="#3b82f6", alpha=0.85)
                ax.set_xlabel("Importance (réduction R²)", color="#9ca3af")
                ax.tick_params(colors="#6b7280")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a2d3e")
                st.pyplot(fig)
                plt.close()

                st.dataframe(df_perm.sort_values("Importance", ascending=False),
                             use_container_width=True)
            except Exception as e:
                st.warning(f"Permutation importance indisponible : {e}")

        with col2:
            st.markdown("#### Feature Importance native")
            if hasattr(model, "feature_importances_"):
                df_fi = pd.DataFrame({
                    "Feature"   : feat_names,
                    "Importance": model.feature_importances_,
                }).sort_values("Importance", ascending=True)

                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor("#0f1117")
                ax.set_facecolor("#1a1d2e")
                ax.barh(df_fi["Feature"], df_fi["Importance"],
                        color="#10b981", alpha=0.85)
                ax.set_xlabel("Importance (Gini)", color="#9ca3af")
                ax.tick_params(colors="#6b7280")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a2d3e")
                st.pyplot(fig)
                plt.close()
            else:
                st.info(f"Pas de feature importance native pour **{best_name}** (modèle non basé sur arbres).")
                st.markdown("Utiliser la **Permutation Importance** à gauche à la place.")

        # Coefficients si régression linéaire
        if hasattr(model, "coef_"):
            st.markdown("#### Coefficients de la Régression Linéaire")
            df_coef = pd.DataFrame({
                "Feature"    : feat_names,
                "Coefficient": model.coef_,
            }).sort_values("Coefficient", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#1a1d2e")
            colors_coef = ["#10b981" if c > 0 else "#ef4444" for c in df_coef["Coefficient"]]
            ax.bar(df_coef["Feature"], df_coef["Coefficient"], color=colors_coef, alpha=0.85)
            ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Coefficient", color="#9ca3af")
            ax.tick_params(colors="#6b7280")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2d3e")
            st.pyplot(fig)
            plt.close()

            st.markdown("""
            **Interprétation** : Chaque coefficient indique l'impact direct d'une feature sur les ventes.
            Un coefficient positif → augmenter ce budget augmente les ventes.
            """)

        # Interprétation business
        st.markdown("---")
        st.markdown("### 💡 Interprétation Business")
        if tache == "🛒 Sales":
            st.markdown("""
            - 📺 **TV** est la variable la plus influente sur les ventes
            - 📻 **Radio** a une contribution significative
            - 📱 **Social Media** a un impact plus faible
            - 🌟 **Influencer** joue un rôle mais moins déterminant que les budgets médias
            - ➡️ Pour maximiser les ventes : **prioriser le budget TV**
            """)
        else:
            st.markdown("""
            - Le **Budget Total** est la variable la plus influente sur le ROI (effet décroissant)
            - Au-delà d'un certain seuil, augmenter le budget **réduit** le ROI
            - 📱 **Social Media** offre souvent le meilleur ROI marginal
            - 🌟 **L'Influencer** n'impacte pas significativement le ROI
            - ➡️ Pour maximiser le ROI : **éviter de sur-investir**, cibler les budgets optimaux
            """)
    else:
        st.warning("⚠️ Modèle non chargé. Exécutez d'abord le notebook `02_prediction_ROI.ipynb` pour générer les fichiers `.pkl`.")
