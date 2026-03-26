import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage de base :
    - copie défensive
    - suppression des espaces dans les noms de colonnes
    - suppression des doublons
    - imputation des valeurs manquantes numériques par la médiane
    """
    df = df.copy()

    # Uniformisation légère des noms de colonnes
    df.columns = [col.strip() for col in df.columns]

    # Suppression des doublons
    df = df.drop_duplicates()

    # Imputation simple des colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def create_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Création de la cible ROI :
    ROI = Sales / (TV + Radio + Social Media)

    Protection contre la division par zéro.
    """
    df = df.copy()

    required_cols = ["TV", "Radio", "Social Media", "Sales"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Colonnes manquantes pour calculer le ROI : {missing_cols}")

    total_budget = df["TV"] + df["Radio"] + df["Social Media"]
    total_budget = total_budget.replace(0, np.nan)

    df["ROI"] = df["Sales"] / total_budget
    df["ROI"] = df["ROI"].fillna(0)

    return df


def handle_outliers(df: pd.DataFrame, roi_upper_cap: float = 10.0) -> pd.DataFrame:
    """
    Limite les valeurs extrêmes du ROI via un capping simple.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant la colonne ROI
    roi_upper_cap : float
        Valeur maximale autorisée pour le ROI

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "ROI" not in df.columns:
        raise ValueError("La colonne 'ROI' est absente. Crée d'abord le ROI avant de traiter les outliers.")

    df["ROI"] = df["ROI"].clip(upper=roi_upper_cap)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodage one-hot de la variable catégorielle 'Influencer'.
    On retire la première modalité pour éviter la redondance.
    """
    df = df.copy()

    if "Influencer" not in df.columns:
        raise ValueError("La colonne 'Influencer' est absente du DataFrame.")

    df = pd.get_dummies(df, columns=["Influencer"], drop_first=True)

    return df


def prepare_features_target(df: pd.DataFrame, target_col: str = "ROI"):
    """
    Sépare les variables explicatives X et la cible y.

    On retire volontairement :
    - ROI : cible
    - Sales : fuite de données, car ROI dépend déjà de Sales
    """
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' est absente.")

    y = df[target_col]
    X = df.drop(columns=[target_col, "Sales"], errors="ignore")

    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Découpage train / test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)