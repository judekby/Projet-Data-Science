def validate_budget_range(tv: float, radio: float, social_media: float) -> None:
    """
    Vérifie que les budgets sont dans des plages cohérentes (en millions).
    Lève une ValueError si une valeur semble aberrante.
    """
    MAX_BUDGET = 1000
    budgets = {"TV": tv, "Radio": radio, "Social Media": social_media}

    for name, value in budgets.items():
        if value < 0:
            raise ValueError(f"Le budget '{name}' ne peut pas être négatif.")
        if value > MAX_BUDGET:
            raise ValueError(
                f"Le budget '{name}' ({value}M€) semble anormalement élevé "
                f"(max accepté : {MAX_BUDGET}M€)."
            )

    if tv == 0 and radio == 0 and social_media == 0:
        raise ValueError("Au moins un budget doit être supérieur à 0.")


def format_currency(value: float, unit: str = "M€") -> str:
    """Formate une valeur en chaîne lisible."""
    return f"{value:,.2f} {unit}"


def format_percentage(value: float) -> str:
    """Formate une rentabilité en pourcentage avec signe."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.0f}%"