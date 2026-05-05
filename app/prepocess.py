import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

INFLUENCER_MAPPING = {"Nano": 0, "Micro": 1, "Macro": 2, "Mega": 3}


def load_data(data_path: str) -> pd.DataFrame:
    """Charge le fichier CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes avec valeurs manquantes et les doublons."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df.copy()


def encode_influencer(df: pd.DataFrame) -> pd.DataFrame:
    """Encode la colonne Influencer avec un mapping ordinal cohérent."""
    if "Influencer" not in df.columns:
        raise ValueError("Colonne 'Influencer' introuvable dans le DataFrame.")
    df["Influencer"] = df["Influencer"].map(INFLUENCER_MAPPING)
    if df["Influencer"].isna().any():
        raise ValueError("Valeurs inconnues dans la colonne 'Influencer'.")
    return df


def scale_to_thousands(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit TV, Radio, Social Media et Sales en milliers."""
    cols = ["TV", "Radio", "Social Media", "Sales"]
    existing = [c for c in cols if c in df.columns]
    df[existing] = (df[existing] * 1000).round(2)
    return df


def compute_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le ROI marketing.
    ROI = Sales / (TV + Radio + Social Media)
    Interprétation : pour 1 unité de budget investi, combien de ventes génère-t-on ?
    Doit être appelé APRÈS scale_to_thousands (toutes les valeurs en milliers).
    """
    required = ["TV", "Radio", "Social Media", "Sales"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' manquante pour calculer le ROI.")

    df["Total_Budget"] = df["TV"] + df["Radio"] + df["Social Media"]

    # Éviter division par zéro
    df["ROI"] = df.apply(
        lambda row: round(row["Sales"] / row["Total_Budget"], 6)
        if row["Total_Budget"] > 0 else 0,
        axis=1
    )

    print(f"💰 ROI calculé — moyenne : {df['ROI'].mean():.4f} | min : {df['ROI'].min():.4f} | max : {df['ROI'].max():.4f}")
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Sauvegarde le DataFrame nettoyé."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Fichier sauvegardé : {output_path}")


def preprocess_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """Pipeline complet de preprocessing."""
    df = load_data(input_path)
    print(f"📥 Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    df = clean_data(df)
    print(f"🧹 Après nettoyage : {df.shape[0]} lignes")

    df = encode_influencer(df)
    print("🔢 Colonne 'Influencer' encodée")

    df = scale_to_thousands(df)
    print("📏 Colonnes converties en milliers")

    df = compute_roi(df)
    print("📊 Colonne ROI ajoutée")

    save_data(df, output_path)
    return df


def prepare_input_for_prediction(data: dict) -> pd.DataFrame:
    """
    Prépare un dictionnaire d'entrée (depuis l'API) en DataFrame prêt pour la prédiction.
    Attendu : {"TV": float, "Radio": float, "Social Media": float, "Influencer": str}
    """
    if "Influencer" not in data:
        raise ValueError("Champ 'Influencer' manquant.")

    influencer_val = INFLUENCER_MAPPING.get(data["Influencer"])
    if influencer_val is None:
        raise ValueError(
            f"Valeur 'Influencer' invalide : '{data['Influencer']}'. "
            f"Valeurs acceptées : {list(INFLUENCER_MAPPING.keys())}"
        )

    tv           = float(data.get("TV", 0)) * 1000
    radio        = float(data.get("Radio", 0)) * 1000
    social_media = float(data.get("Social Media", 0)) * 1000

    df = pd.DataFrame([{
        "TV"          : tv,
        "Radio"       : radio,
        "Social Media": social_media,
        "Influencer"  : influencer_val,
    }])
    return df


def compute_roi_from_input(tv: float, radio: float, social_media: float, predicted_sales: float) -> float:
    """
    Calcule le ROI à partir des budgets en millions et des ventes prédites en milliers.
    Utilisé par l'API après une prédiction.
    """
    total_budget = (tv + radio + social_media) * 1000  # conversion en milliers
    if total_budget == 0:
        return 0.0
    return round(predicted_sales / total_budget, 6)


if __name__ == "__main__":
    input_path  = "data/raw/marketing_sales.csv"
    output_path = "data/processed/marketing_and_sales_clean.csv"
    preprocess_pipeline(input_path, output_path)
