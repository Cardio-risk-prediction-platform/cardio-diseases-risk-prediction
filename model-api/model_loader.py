import pickle
import pandas as pd

# Chargement du modèle au démarrage
with open("cardio_model.pkl", "rb") as f:
    package = pickle.load(f)

model = package["model"]
scaler = package["scaler"]
expected_features = package["features"]


def predict_disease(data: dict):
    # Renommer les colonnes physiques
    data["Height_(cm)"] = data.pop("Height_cm", None)
    data["Weight_(kg)"] = data.pop("Weight_kg", None)

    # Convertir en DataFrame
    df = pd.DataFrame([data])
    
    # Encodage des colonnes catégorielles
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ajouter toutes les colonnes attendues et mettre à 0 si manquantes
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Réordonner les colonnes exactement comme à l'entraînement
    df_encoded = df_encoded[expected_features]

    # Normaliser
    df_scaled = scaler.transform(df_encoded)
    
    # Prédiction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][list(model.classes_).index("Yes")]

    return prediction, probability