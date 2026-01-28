import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Predictive Maintenance API", description="AI Service to detect machine failures")

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 2. Load the artifacts (Ensure these files exist in the /models folder!)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

app = FastAPI()

# 3. Define the data format once
class MachineData(BaseModel):
    # Field allows you to set default values and descriptions for the UI
    machine_type: str = Field(..., example="L", description="Type of machine: L (Low), M (Medium), or H (High)")
    air_temperature: float = Field(..., example=298.1, description="Air temperature in Kelvin")
    process_temperature: float = Field(..., example=308.6, description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., example=1551, description="Rotational speed in RPM")
    torque: float = Field(..., example=42.8, description="Torque in Nm")
    tool_wear: int = Field(..., example=0, description="Tool wear in minutes")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_failure(data: MachineData):
    # prediction logic
    type_encoded = le.transform([data.machine_type])[0]
    input_data = [[type_encoded, data.air_temperature, data.process_temperature, 
                    data.rotational_speed, data.torque, data.tool_wear]]
    
    prediction = model.predict(input_data)[0]
    # confidence score
    probability = model.predict_proba(input_data)[0].max()
    
    return {
        "status": "Success",
        "failure_predicted": "Yes" if prediction == 1 else "No",
        "confidence": f"{round(probability * 100, 2)}%"
    }