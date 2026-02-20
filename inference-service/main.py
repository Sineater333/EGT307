# main.py (Inference Service)
from datetime import datetime, timezone
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Predictive Maintenance API",
    description="AI Service to detect machine failures",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_model.pkl"))
type_model = joblib.load(os.path.join(MODEL_DIR, "type_model.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

class MachineData(BaseModel):
    machine_type: str = Field(..., example="L", description="Type of machine: L, M, or H")
    air_temperature: float = Field(..., example=298.1, description="Air temperature in Kelvin")
    process_temperature: float = Field(..., example=308.6, description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., example=1551, description="Rotational speed in RPM")
    torque: float = Field(..., example=42.8, description="Torque in Nm")
    tool_wear: int = Field(..., example=0, description="Tool wear in minutes")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_failure(data: MachineData):
    type_encoded = le.transform([data.machine_type])[0]
    features_df = pd.DataFrame([{
        "Type": type_encoded,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    is_failing = int(binary_model.predict(features_df)[0])
    cause = None
    if is_failing == 1:
        cause = type_model.predict(features_df)[0]

    # Use a stable timestamp format (ISO, UTC)
    ts = datetime.now(timezone.utc).isoformat()

    return {
        "machine_type": data.machine_type,
        "air_temperature": data.air_temperature,
        "process_temperature": data.process_temperature,
        "rotational_speed": data.rotational_speed,
        "torque": data.torque,
        "tool_wear": data.tool_wear,
        "is_failure": bool(is_failing),
        "status": "Failure Detected" if is_failing else "Healthy",
        "failure_cause": cause,   # None when healthy (not "None" string)
        "timestamp": ts
    }