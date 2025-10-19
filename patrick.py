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
@app.post("/predict/html", response_class=HTMLResponse)
def predict_car_price_html(features: CarFeatures):
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

        html = f"""
        <table border="1">
            <tr><th>Feature</th><th>Value</th></tr>
            <tr><td>Manufacturer</td><td>{manufacturer}</td></tr>
            <tr><td>Model</td><td>{model_name}</td></tr>
            <tr><td>Fuel type</td><td>{fuel}</td></tr>
            <tr><td>Engine size</td><td>{engine}</td></tr>
            <tr><td>Year of manufacture</td><td>{year}</td></tr>
            <tr><td>Mileage</td><td>{mileage}</td></tr>
            <tr><td>Age</td><td>{age}</td></tr>
            <tr><td>Mileage per year</td><td>{mileage_per_year:.2f}</td></tr>
            <tr><td>Vintage</td><td>{vintage}</td></tr>
            <tr><td>Predicted Price (GBP)</td><td>{prediction:.2f}</td></tr>
        </table>
        """
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run locally:
# uvicorn patrick:app --reload