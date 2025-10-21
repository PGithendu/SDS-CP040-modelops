import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

# Load the pre-trained model
model = joblib.load("model.pkl")  # <-- FIXED PATH

app = FastAPI(title="Car Price Prediction API", version="1.0.0")

# Input schema for car features
class CarFeatures(BaseModel):
    Manufacturer: str
    Model: str
    Fuel_type: str
    Engine_size: float
    Year_of_manufacture: int
    Mileage: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/metadata")
def get_metadata():
    return {
        "model_info": "Car Price Prediction Model",
        "model": "model.pkl",
        "version": "1.0.0",
        "features": [
            "Manufacturer", "Model", "Engine_size", "Fuel_type", 
            "Year_of_manufacture", "Mileage"
        ]
    }

@app.post("/predict")
def predict_car_price(features: CarFeatures):
    try:
        manufacturer = features.Manufacturer.strip()
        model_name = features.Model.strip()
        fuel = features.Fuel_type.strip()
        engine = float(features.Engine_size)
        year = int(features.Year_of_manufacture)
        mileage = float(features.Mileage)

        CURRENT_YEAR = 2025
        age = max(CURRENT_YEAR - year, 0)
        mileage_per_year = mileage / max(age, 1)
        vintage = int(age >= 20)

        row = {
            "Manufacturer": manufacturer,
            "Model": model_name,
            "Fuel type": fuel,
            "Engine size": engine,
            "Year of manufacture": year,
            "Mileage": mileage,
            "age": age,
            "mileage_per_year": mileage_per_year,
            "vintage": vintage,
        }
        df = pd.DataFrame([row])
        prediction = model.predict(df)[0]

        return {"predicted_price_gbp": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint to serve index.html from the project root
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# To run locally:
# uvicorn patrick:app --reload