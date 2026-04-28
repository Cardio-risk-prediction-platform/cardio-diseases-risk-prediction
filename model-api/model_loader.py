import pickle
import pandas as pd

# Chargement du modèle au démarrage
with open("cardio_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_disease(data: dict):
    # mapping noms colonnes
    data["Height_(cm)"] = data.pop("Height_cm", None)
    data["Weight_(kg)"] = data.pop("Weight_kg", None)

    # Convertir en DataFrame
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][list(model.classes_).index("Yes")]

    return prediction, probability