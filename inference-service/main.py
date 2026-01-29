from datetime import datetime
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Predictive Maintenance API", description="AI Service to detect machine failures")

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Define CSV Log Path
LOG_FILE = os.path.join(BASE_DIR, "logs", "prediction_history.csv")
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

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
def predict_failure(data: MachineData):
    # prediction logic
    type_encoded = le.transform([data.machine_type])[0]
    input_data = [[type_encoded, data.air_temperature, data.process_temperature, 
                    data.rotational_speed, data.torque, data.tool_wear]]
    
    # 1. Check if broken
    is_failing = binary_model.predict(input_data)[0]

    cause = "None"
    if is_failing == 1:
        # 2. If broken, ask the second model why
        cause = type_model.predict(input_data)[0]
    
    result = {
        "status": "Healthy" if is_failing == 0 else "Failure Detected",
        "failure_cause": cause,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 3. Logging Logic
    # This history will be for our dashboard
    df_log = pd.DataFrame([{**data.dict(), **result}])
    df_log.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

    return result