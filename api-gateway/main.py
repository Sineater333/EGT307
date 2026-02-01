from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Engineering API Gateway")

# Internal Kubernetes URL for the inference service
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_URL", "http://inference-service:8000/predict")

class MachineData(BaseModel):
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int

@app.get("/health")
def health_check():
    return {"status": "Gateway is online"}

@app.post("/route/predict")
async def route_prediction(data: MachineData):
    async with httpx.AsyncClient() as client:
        try:
            # Forward the request to the internal Inference Service
            response = await client.post(INFERENCE_SERVICE_URL, json=data.dict())
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Inference Service Error")
            
            return response.json()
        
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Inference Service is unreachable")