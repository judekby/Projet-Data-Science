import joblib
import os
from typing import Any, Optional

# ── Chemins par défaut ────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

SALES_MODEL_PATH    = os.path.join(BASE_DIR, "model", "best_model.pkl")
SALES_PREPROC_PATH  = os.path.join(BASE_DIR, "model", "preprocessor.pkl")
ROI_MODEL_PATH      = os.path.join(BASE_DIR, "model", "best_model_roi.pkl")
ROI_PREPROC_PATH    = os.path.join(BASE_DIR, "model", "preprocessor_roi.pkl")

# ── Cache global ──────────────────────────────────────────────
_models = {
    "sales_model"   : None,
    "sales_preproc" : None,
    "roi_model"     : None,
    "roi_preproc"   : None,
}


def _load_pkl(path: str, label: str) -> Optional[Any]:
    """Charge un fichier .pkl et retourne None si introuvable."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print(f"⚠️  {label} introuvable : {abs_path}")
        return None
    obj = joblib.load(abs_path)
    print(f"✅ {label} chargé depuis : {abs_path}")
    return obj


def load_models() -> dict:
    """Charge tous les modèles et préprocesseurs au démarrage."""
    _models["sales_model"]   = _load_pkl(SALES_MODEL_PATH,   "Modèle Sales")
    _models["sales_preproc"] = _load_pkl(SALES_PREPROC_PATH, "Preprocessor Sales")
    _models["roi_model"]     = _load_pkl(ROI_MODEL_PATH,     "Modèle ROI")
    _models["roi_preproc"]   = _load_pkl(ROI_PREPROC_PATH,   "Preprocessor ROI")
    return _models


def get_sales_model() -> Any:
    if _models["sales_model"] is None:
        raise RuntimeError("Modèle Sales non chargé.")
    return _models["sales_model"]


def get_sales_preproc() -> Any:
    if _models["sales_preproc"] is None:
        raise RuntimeError("Preprocessor Sales non chargé.")
    return _models["sales_preproc"]


def get_roi_model() -> Any:
    if _models["roi_model"] is None:
        raise RuntimeError("Modèle ROI non chargé.")
    return _models["roi_model"]


def get_roi_preproc() -> Any:
    if _models["roi_preproc"] is None:
        raise RuntimeError("Preprocessor ROI non chargé.")
    return _models["roi_preproc"]


def is_sales_loaded() -> bool:
    return _models["sales_model"] is not None and _models["sales_preproc"] is not None


def is_roi_loaded() -> bool:
    return _models["roi_model"] is not None and _models["roi_preproc"] is not None


def get_sales_model_name() -> str:
    return type(_models["sales_model"]).__name__ if _models["sales_model"] else "N/A"


def get_roi_model_name() -> str:
    return type(_models["roi_model"]).__name__ if _models["roi_model"] else "N/A"