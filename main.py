from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from dotenv import load_dotenv
import os

# 1. Load local agent configuration thresholds
load_dotenv()
CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", 0.85))

app = FastAPI(title="KrishiAI Sentinel - ML/DL Inference Core")

# 2. Define the structural matrix input schema for IoT Telemetry
class TelemetryPayload(BaseModel):
    ambient_temperature: float
    soil_moisture: float
    uav_altitude: float
    sensor_reading_array: list[float]  # Sequence for temporal processing

# 3. Simulated Model Class to represent your PyTorch Weights
class ProactiveHazardModel:
    def __init__(self):
        # Determine if local hardware supports GPU acceleration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SYSTEM] Loading deep learning weights onto computational device: {self.device}")
        
    def forward_inference(self, data_tensor: torch.Tensor) -> float:
        # In a real setup, this runs: output = self.model(data_tensor)
        # Simulating a matrix calculation outputting a hazard probability score
        return float(torch.sigmoid(torch.mean(data_tensor)).item())

# Initialize the research model engine
hazard_engine = ProactiveHazardModel()

# ==========================================
# ML/DL ROUTERS
# ==========================================

@app.get("/engine/status")
def get_engine_status():
    """Returns the computational health of the DL platform."""
    return {
        "status": "operational",
        "device_allocated": hazard_engine.device,
        "agent_gate_threshold": CONFIDENCE_THRESHOLD
    }

@app.post("/inference/spatial-temporal")
def process_telemetry(payload: TelemetryPayload):
    """
    ML/DL Router: Ingests telemetry, converts to tensor, 
    runs forward inference, and applies agent decision logic.
    """
    try:
        # Step 1: Convert incoming sequence array into a PyTorch Tensor
        raw_data = payload.sensor_reading_array
        data_tensor = torch.tensor(raw_data, dtype=torch.float32)
        
        # Step 2: Pass tensor through the forward inference pass
        hazard_probability = hazard_engine.forward_inference(data_tensor)
        
        # Step 3: Agent Decision Gate Logic
        action_required = hazard_probability >= CONFIDENCE_THRESHOLD
        
        return {
            "inference_metrics": {
                "hazard_probability": round(hazard_probability, 4),
                "threshold_passed": action_required
            },
            "agent_decision": {
                "status": "CRITICAL_ALERT" if action_required else "NOMINAL_MONITORING",
                "instruction": "Dispatch drone mitigation protocol" if action_required else "Continue loop"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))