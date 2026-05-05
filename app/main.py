from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.schemas import (
    PredictRequest, PredictResponse,
    HealthResponse, ModelInfoResponse
)
from app.predict import run_prediction
from app.model_loader import (
    load_models,
    is_sales_loaded, is_roi_loaded,
    get_sales_model_name, get_roi_model_name,
)
from app.utils import validate_budget_range


# ── Lifespan : chargement au démarrage ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles une seule fois au démarrage de l'API."""
    print("🚀 Démarrage de l'API Marketing ROI...")
    load_models()
    if is_sales_loaded() and is_roi_loaded():
        print("✅ Tous les modèles sont chargés.")
    else:
        print("⚠️  Certains modèles ne sont pas chargés.")
        print("   Lancez les notebooks pour générer les .pkl manquants.")
    yield
    print("🛑 Arrêt de l'API.")


# ── Application FastAPI ───────────────────────────────────────
app = FastAPI(
    title="Marketing ROI Prediction API",
    description=(
        "API de prédiction des ventes et du ROI marketing à partir des budgets "
        "TV, Radio, Social Media et du type d'influenceur.\n\n"
        "**Projet Data Science M2 — EFREI 2025-26**\n\n"
        "Tâche principale : Prédiction du ROI (retour sur investissement)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS pour permettre l'appel depuis le dashboard Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    """Point d'entrée racine."""
    return {
        "name"   : "Marketing ROI Prediction API",
        "version": "1.0.0",
        "docs"   : "/docs",
        "health" : "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Vérifie l'état de l'API et des modèles."""
    sales_ok = is_sales_loaded()
    roi_ok   = is_roi_loaded()
    status   = "ok" if (sales_ok and roi_ok) else "degraded"

    return HealthResponse(
        status             = status,
        sales_model_loaded = sales_ok,
        roi_model_loaded   = roi_ok,
        sales_model_name   = get_sales_model_name(),
        roi_model_name     = get_roi_model_name(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Retourne les informations sur les modèles déployés."""
    if not is_sales_loaded() or not is_roi_loaded():
        raise HTTPException(status_code=503, detail="Modèles non disponibles.")

    return ModelInfoResponse(
        sales_model = get_sales_model_name(),
        roi_model   = get_roi_model_name(),
        features    = ["TV", "Radio", "Social Media", "Influencer"],
        targets     = ["Sales (en milliers)", "ROI (ratio)"],
        description = (
            "Deux modèles supervisés entraînés pour prédire les ventes et le ROI "
            "à partir des budgets marketing multicanal. "
            "Le ROI est calculé comme Sales / (TV + Radio + Social Media)."
        ),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
def predict(request: PredictRequest):
    """
    Prédit les ventes (k€ et M€) et le ROI à partir des budgets marketing.

    **Entrée JSON attendue :**
    ```json
    {
        "TV": 2.0,
        "Radio": 1.0,
        "Social Media": 0.5,
        "Influencer": "Macro"
    }
    ```

    **Les budgets sont en millions d'euros.**

    **Réponse :**
    - `predicted_sales_k` : ventes prédites en milliers
    - `predicted_sales_m` : ventes prédites en millions
    - `predicted_roi` : ROI prédit (ratio brut, ex: 2.62)
    - `roi_percentage` : rentabilité en % (ex: +162%)
    - `total_budget_m` : budget total investi en millions
    - `gain_net_m` : gain net estimé (Ventes - Budget)
    """
    # Vérification du chargement
    if not is_sales_loaded() or not is_roi_loaded():
        raise HTTPException(
            status_code=503,
            detail="Un ou plusieurs modèles ne sont pas chargés. Vérifiez /health."
        )

    # Validation métier
    try:
        validate_budget_range(request.TV, request.Radio, request.Social_Media)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Prédiction
    try:
        result = run_prediction(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

    return result
