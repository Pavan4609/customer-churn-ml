from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/model.pkl")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0"
)


# Input data
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Customer may churn"
    else:
        result = "Customer likely to stay"

    return {
        "prediction": int(prediction),
        "result": result,
        "churn_probability": round(float(probability), 2)
    }