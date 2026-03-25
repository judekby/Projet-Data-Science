import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Charge les données brutes depuis un fichier Excel ou CSV.

    Parameters
    ----------
    path : str
        Chemin vers le fichier de données.

    Returns
    -------
    pd.DataFrame
        Données chargées dans un DataFrame pandas.
    """
    if path.endswith(".xls") or path.endswith(".xlsx"):
        df = pd.read_excel(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Format de fichier non supporté. Utilise .xls, .xlsx ou .csv")

    return df