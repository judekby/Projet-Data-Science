import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage de base :
    - suppression des doublons
    - standardisation légère des noms de colonnes
    - gestion simple des valeurs manquantes
    """
    df = df.copy()

    # Uniformiser légèrement les noms de colonnes
    df.columns = [col.strip() for col in df.columns]

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Remplissage simple des valeurs manquantes sur les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def create_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la colonne ROI simplifiée :
    ROI = Sales / (TV + Radio + Social Media)

    Une protection est appliquée contre la division par zéro.
    """
    df = df.copy()

    required_cols = ["TV", "Radio", "Social Media", "Sales"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Colonnes manquantes pour calculer le ROI : {missing_cols}")

    total_budget = df["TV"] + df["Radio"] + df["Social Media"]
    total_budget = total_budget.replace(0, np.nan)

    df["ROI"] = df["Sales"] / total_budget

    # Option simple pour éviter les NaN liés à budget = 0
    df["ROI"] = df["ROI"].fillna(0)

    return df