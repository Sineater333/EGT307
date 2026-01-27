import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 2. Load the artifacts (Ensure these files exist in the /models folder!)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

app = FastAPI()

# 3. Define the data format once
class MachineData(BaseModel):
    machine_type: str  # Expects 'L', 'M', or 'H'
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_failure(data: MachineData):
    # Convert 'L/M/H' to numerical value
    type_encoded = le.transform([data.machine_type])[0]
    
    # Create the input list in the exact order used during training
    input_data = [[
        type_encoded, 
        data.air_temperature, 
        data.process_temperature, 
        data.rotational_speed, 
        data.torque, 
        data.tool_wear
    ]]
    
    prediction = model.predict(input_data)
    
    # Get probability if available
    probability = 1.0
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data).max()
    
    return {
        "failure_predicted": int(prediction[0]),
        "confidence": float(probability)
    }