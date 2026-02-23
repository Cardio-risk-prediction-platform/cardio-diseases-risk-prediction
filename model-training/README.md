

# 📄 **README — Export du Meilleur Modèle (Machine Learning)**

## 🎯 Objectif
Après avoir entraîné plusieurs modèles de prédiction du risque cardiovasculaire (RandomForest, Logistic Regression), nous avons sélectionné **le meilleur modèle** selon des métriques adaptées au contexte médical (Recall, F1-score, ROC AUC).
L’objectif de cette étape est d’exporter ce modèle afin qu’il puisse être utilisé dans l’API de prédiction.

---
## 🧠 Pourquoi exporter le modèle ?
Le notebook d’entraînement sert uniquement à :
- charger les données
- préparer les features
- entraîner plusieurs modèles
- comparer leurs performances

Mais l’API (FastAPI) a besoin d’un **fichier modèle** pour faire des prédictions sur de nouvelles données.
Ce fichier doit contenir :
- le modèle entraîné
- le scaler utilisé pour normaliser les données
- la liste des features dans le bon ordre
C’est ce que nous appelons un **model package**.
---

## 🏆 Sélection du meilleur modèle
Après comparaison des performances :

- **RandomForest (balanced)** → accuracy élevée mais recall très faible
- **Logistic Regression (balanced)** → recall élevé, F1-score meilleur, ROC AUC supérieur

Dans un contexte médical, la priorité est de **détecter les patients à risque**.
Nous avons donc choisi :

### ✔️ **Modèle final : Logistic Regression (class_weight="balanced")**

---

## 📦 Construction du “model_package”
Pour que l’API puisse utiliser le modèle, nous avons regroupé dans un dictionnaire Python :

- `model` → le modèle entraîné
- `scaler` → l’objet StandardScaler utilisé pour normaliser les données
- `features` → la liste des colonnes utilisées pour l’entraînement

Ce package garantit que l’API reproduira exactement les mêmes transformations que lors de l’entraînement.

---

## 💾 Export du modèle au format `.pkl`

Voici le code utilisé pour exporter le modèle :

```python
# ============================
# 14. EXPORT DU MEILLEUR MODÈLE
# ============================

import pickle

# Sélection du meilleur modèle selon les métriques
best_model = log_reg

# Construction du package contenant tout ce dont l'API a besoin
model_package = {
    "model": best_model,                     # modèle final entraîné
    "scaler": scaler,                        # scaler utilisé pour normaliser les données
    "features": X_scaled.columns.tolist()    # liste des colonnes dans le bon ordre
}

# Sauvegarde du package dans un fichier pickle
with open("cardio_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("Modèle sauvegardé dans cardio_model.pkl")
```

---

## 📁 Où placer le fichier exporté ?
Pour respecter une architecture propre :

```
model_training/
    training.ipynb

model_api/
    cardio_model.pkl   ← fichier exporté ici
    model.py
    requirements.txt
```

- Le notebook reste dans `model_training/`
- Le modèle `.pkl` est placé dans `model_api/` car c’est là que l’API le chargera

---

## 🚀 Étape suivante
L’API FastAPI pourra maintenant :

1. charger `cardio_model.pkl`
2. appliquer le scaler
3. préparer les données dans le bon ordre
4. exécuter `model.predict()`
5. renvoyer une prédiction fiable

