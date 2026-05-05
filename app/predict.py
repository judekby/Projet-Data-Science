import pandas as pd

from app.schemas import PredictRequest, PredictResponse
from app.model_loader import (
    get_sales_model, get_sales_preproc,
    get_roi_model,   get_roi_preproc,
    get_sales_model_name, get_roi_model_name,
    is_sales_loaded, is_roi_loaded,
)

INFLUENCER_MAPPING = {"Nano": 0, "Micro": 1, "Macro": 2, "Mega": 3}


def _build_input_dataframe(request: PredictRequest) -> pd.DataFrame:
    """
    Construit le DataFrame d'entrée pour les modèles.
    Les budgets sont en millions dans la requête → conversion en milliers ici.
    """
    return pd.DataFrame([{
        "TV"                : request.TV * 1000,
        "Radio"             : request.Radio * 1000,
        "Social Media"      : request.Social_Media * 1000,
        "Influencer_encoded": INFLUENCER_MAPPING[request.Influencer],
    }])


def run_prediction(request: PredictRequest) -> PredictResponse:
    """
    Exécute la prédiction Sales + ROI à partir des données de la requête.

    Étapes :
    1. Construire le DataFrame d'entrée
    2. Préprocesser et prédire avec le modèle Sales
    3. Préprocesser et prédire avec le modèle ROI
    4. Calculer les métriques dérivées (gain net, rentabilité %)
    5. Retourner la réponse unifiée
    """
    if not is_sales_loaded() or not is_roi_loaded():
        raise RuntimeError("Tous les modèles ne sont pas chargés.")

    # 1. DataFrame d'entrée
    input_df = _build_input_dataframe(request)

    # 2. Prédiction Sales (en milliers)
    sales_input = get_sales_preproc().transform(input_df)
    pred_sales_k = float(get_sales_model().predict(sales_input)[0])

    # 3. Prédiction ROI (ratio)
    roi_input = get_roi_preproc().transform(input_df)
    pred_roi = float(get_roi_model().predict(roi_input)[0])

    # 4. Calculs dérivés
    total_budget_m  = request.TV + request.Radio + request.Social_Media
    pred_sales_m    = pred_sales_k / 1000  # k€ → M€
    gain_net_m      = pred_sales_m - total_budget_m
    roi_percentage  = (pred_roi - 1) * 100

    return PredictResponse(
        predicted_sales_m  = round(pred_sales_m, 4),
        roi_percentage     = round(roi_percentage, 2),
        total_budget_m     = round(total_budget_m, 2),
        gain_net_m         = round(gain_net_m, 4),
        sales_model        = get_sales_model_name(),
        roi_model          = get_roi_model_name(),
        influencer_encoded = INFLUENCER_MAPPING[request.Influencer],
    )