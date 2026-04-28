import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Charger le dataset
df = pd.read_csv("../model-training/CVD_cleaned.csv")

# Target
y = df["Heart_Disease"]
X = df.drop(columns=["Heart_Disease"])

# Colonnes
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# Pipeline complet
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entraînement
pipeline.fit(X_train, y_train)

# Sauvegarde
with open("cardio_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Modèle entraîné et sauvegardé")