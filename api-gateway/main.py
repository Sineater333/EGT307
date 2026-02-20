from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Engineering API Gateway")

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference-service:8000/predict")
DB_LOG_URL = os.getenv("DB_LOG_URL", "http://database-service:8000/logs")
DB_HISTORY_URL = os.getenv("DB_HISTORY_URL", "http://database-service:8000/history")

class MachineData(BaseModel):
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int

@app.get("/health")
async def health_check():
    # Gateway is healthy if it can respond.
    # Optional:
    return {"status": "healthy", "service": "api-gateway"}

@app.post("/predict")
async def predict(data: MachineData):
    timeout = httpx.Timeout(5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # 1) Call inference
        try:
            inf_resp = await client.post(INFERENCE_URL, json=data.model_dump())
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Inference Service is unreachable")

        if inf_resp.status_code != 200:
            raise HTTPException(status_code=inf_resp.status_code, detail="Inference Service Error")

        result = inf_resp.json()

        # 2) log to DB (do NOT fail predict if DB is down)
        try:
            await client.post(DB_LOG_URL, json=result)
        except Exception:
            # swallow logging errors
            pass

        return result

@app.get("/history")
async def history():
    timeout = httpx.Timeout(5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            db_resp = await client.get(DB_HISTORY_URL)
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Database Service is unreachable")

        if db_resp.status_code != 200:
            raise HTTPException(status_code=db_resp.status_code, detail="Database Service Error")

        return db_resp.json()