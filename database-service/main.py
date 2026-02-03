from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import os

app = FastAPI(title="Maintenance History Service")

# MongoDB connection string - using K8s service name 'mongodb'
MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongodb:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.predictive_maintenance

class PredictionLog(BaseModel):
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    status: str
    failure_cause: str
    timestamp: str

@app.post("/logs")
async def save_log(log: PredictionLog):
    result = await db.history.insert_one(log.dict())
    if result.inserted_id:
        return {"status": "Log saved", "id": str(result.inserted_id)}
    raise HTTPException(status_code=500, detail="Failed to save log")

@app.get("/history")
async def get_history():
    cursor = db.history.find().sort("timestamp", -1).limit(100)
    history = await cursor.to_list(length=100)
    for h in history:
        h["_id"] = str(h["_id"])
    return history

@app.get("/health")
async def health_check():
    try:
        await db.command("ping")
        return {"status": "healthy", "database": "connected"}
    except Exception:
        return {"status": "unhealthy", "database": "disconnected"}