from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    """Schéma d'entrée pour la prédiction (Sales et ROI)."""

    TV: float = Field(..., ge=0, description="Budget TV en millions (ex: 2.0)")
    Radio: float = Field(..., ge=0, description="Budget Radio en millions (ex: 1.0)")
    Social_Media: float = Field(
        ..., ge=0, alias="Social Media",
        description="Budget Social Media en millions (ex: 0.5)"
    )
    Influencer: Literal["Mega", "Macro", "Micro", "Nano"] = Field(
        ..., description="Type d'influenceur mobilisé"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "TV": 2.0,
                "Radio": 1.0,
                "Social Media": 0.5,
                "Influencer": "Macro"
            }
        }


class PredictResponse(BaseModel):
    """Schéma de sortie unifié — Sales + ROI."""

    predicted_sales_k: float = Field(..., description="Ventes prédites en milliers d'euros")
    predicted_sales_m: float = Field(..., description="Ventes prédites en millions d'euros")
    predicted_roi: float = Field(..., description="ROI prédit (ratio brut, ex: 2.62)")
    roi_percentage: float = Field(..., description="Rentabilité en pourcentage (ex: +162%)")

    total_budget_m: float = Field(..., description="Budget total investi en millions")
    gain_net_m: float = Field(..., description="Gain net estimé en millions (Ventes - Budget)")

    sales_model: str = Field(..., description="Nom du modèle Sales utilisé")
    roi_model: str = Field(..., description="Nom du modèle ROI utilisé")

    influencer_encoded: int = Field(..., description="Valeur encodée de l'influenceur (0 à 3)")


class HealthResponse(BaseModel):
    """Réponse du endpoint /health."""

    status: str
    sales_model_loaded: bool
    roi_model_loaded: bool
    sales_model_name: str
    roi_model_name: str


class ModelInfoResponse(BaseModel):
    """Réponse du endpoint /model-info."""

    sales_model: str
    roi_model: str
    features: list[str]
    targets: list[str]
    description: str