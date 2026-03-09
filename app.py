from fastapi import FastAPI
import joblib
import pandas as pd

# Create FastAPI instance
app = FastAPI()

# Load trained model
model = joblib.load("car_price_model.pkl")

# Root endpoint
@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(
    Year: int,
    Present_Price: float,
    Kms_Driven: int,
    Fuel_Type: int,
    Seller_Type: int,
    Transmission: int,
    Owner: int
):

    # Create dataframe from input
    data = pd.DataFrame([{
        "Year": Year,
        "Present_Price": Present_Price,
        "Kms_Driven": Kms_Driven,
        "Fuel_Type": Fuel_Type,
        "Seller_Type": Seller_Type,
        "Transmission": Transmission,
        "Owner": Owner
    }])

    # Make prediction
    prediction = model.predict(data)

    return {
        "predicted_price": float(prediction[0])
    }