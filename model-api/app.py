from fastapi import FastAPI
from schemas import PatientData
from model_loader import predict_disease

app = FastAPI(title="Cardiovascular Diseases Risk Prediction API")

@app.get("/")
def root():
    return {"message": "API Cardio Health Risk Running! Click here to see Swagger documentation http://127.0.0.1:8000/docs"}

@app.post("/predict")
def predict_route(patient: PatientData):

    prediction, probability = predict_disease(patient.dict())

    risk_level = (
        "Low"
        if probability < 0.3
        else "Medium"
        if probability < 0.7
        else "High"
    )

    return {
        "prediction": prediction,
        "probability": round(float(probability), 3),
        "risk_level": risk_level
    }