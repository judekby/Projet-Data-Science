"""
Marketing ROI Prediction API
============================
API REST de prédiction des ventes et du ROI marketing.

Modules :
- preprocess     : pipeline de nettoyage et transformation des données
- schemas        : modèles Pydantic pour validation entrée/sortie
- model_loader   : chargement des modèles sauvegardés (.pkl)
- predict        : logique de prédiction
- utils          : fonctions utilitaires
- main           : point d'entrée FastAPI
"""

__version__ = "1.0.0"