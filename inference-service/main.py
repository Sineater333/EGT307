from datetime import datetime
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import httpx

app = FastAPI(title="Predictive Maintenance API", description="AI Service to detect machine failures")

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Define Database Service URL
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://database-service:8000/logs")

# 2. Load the artifacts (Ensure these files exist in the /models folder!)
binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_model.pkl"))
type_model = joblib.load(os.path.join(MODEL_DIR, "type_model.pkl"))
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
async def predict_failure(data: MachineData):
    # prediction logic
    type_encoded = le.transform([data.machine_type])[0]
    features_df = pd.DataFrame([{
        "Type": type_encoded,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])
    
    # 1. Check if broken
    is_failing = int(binary_model.predict(features_df)[0])

    cause = "None"
    if is_failing == 1:
        # 2. If broken, ask the second model why
        cause = type_model.predict(features_df)[0]
    
    result = {
        "machine_type": data.machine_type,
        "air_temperature": data.air_temperature,
        "process_temperature": data.process_temperature,
        "rotational_speed": data.rotational_speed,
        "torque": data.torque,
        "tool_wear": data.tool_wear,
        "status": "Healthy" if is_failing == 0 else "Failure Detected",
        "failure_cause": cause,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    async with httpx.AsyncClient() as client:
        try:
            db_response = await client.post(DB_SERVICE_URL, json=result, timeout=5.0)
            if db_response.status_code == 200:
                print(f"✅ Successfully logged to DB: {db_response.json()}")
            else:
                # This will print the EXACT reason the DB rejected the data
                print(f"❌ DB Service rejected data (Status {db_response.status_code}): {db_response.text}")
        except Exception as e:
            print(f"⚠️ Failed to connect to database: {e}")
    
    return result