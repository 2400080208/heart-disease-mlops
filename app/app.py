from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Heart Disease Prediction API")

project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "heart_disease_model.joblib"

model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}