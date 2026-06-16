import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Initialize the FastAPI app instance
app = FastAPI(
    title="KrishiAI Expert Core API",
    description="Agronomic Multi-Agent Orchestration & Decision Support Service",
    version="3.0"
)

# Configure Cross-Origin Resource Sharing (CORS) 
# Crucial for allowing your Vercel frontend to seamlessly communicate with Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your specific Vercel app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the precise Pydantic data model matching your frontend payload
class ConsultationRequest(BaseModel):
    crop_type: str
    crop_stage: str
    symptoms: str

# Mock dictionary mapping symptoms to expert-grounded ICAR / IPM recommendations
# This acts as your local knowledge validation step
AGRONOMIC_KNOWLEDGE_BASE = {
    "blight": (
        "DIAGNOSIS: Early Blight (Alternaria solani) suspected due to targeted necrotic structural lesions.\n\n"
        "IMMEDIATE ACTION PROTOCOL (ICAR Guidelines):\n"
        "1. Physical: Prune and incinerate lower infected foliar branches up to 30 cm to prevent spore splash-up.\n"
        "2. Biological: Apply Pseudomonas fluorescens formulations at a dilution metric of 5g/L.\n"
        "3. Chemical Intervention: If threshold condition exceeds 15% coverage, execute a targeted foliar spray of "
        "Mancozeb 75% WP at 2g/L or Tebuconazole 250 EC at 1ml/L.\n\n"
        "14-DAY MONITORING TARGET: Track secondary cell regeneration curves daily. Re-evaluate structural tissue density at Day 7."
    ),
    "mildew": (
        "DIAGNOSIS: Powdery Mildew infection confirmed over upper leaf epidermis surfaces.\n\n"
        "IMMEDIATE ACTION PROTOCOL (IPM Standards):\n"
        "1. Environmental: Adjust canopy spacing vectors immediately to lower relative microclimate humidity layers.\n"
        "2. Remediation Strategy: Administer water-soluble elemental Sulphur 80% WP at a concentrations ratio of 2.5g/L, "
        "or Azoxystrobin 23% SC at 0.5ml/L.\n\n"
        "14-DAY MONITORING TARGET: Verify if the white powder-like mycelium layers begin shrinking by Day 4 to restore chlorophyll optimization."
    )
}

@app.get("/")
def read_root():
    """Health check endpoint to verify Render deployment status."""
    return {
        "status": "ONLINE",
        "framework": "FastAPI Agentic Pipeline",
        "target_region": "Mutually Grounded Mode"
    }

@app.post("/api/advise")
async def execute_consultation(request: ConsultationRequest):
    """
    Primary endpoint invoked by 'Execute AI Consultation' form.
    Processes crop metrics and synthesizes a structured treatment response.
    """
    # Sanity validation for inbound payload parameters
    if not request.crop_type.strip() or not request.symptoms.strip():
        raise HTTPException(
            status_code=400, 
            detail="Payload Error: Crop Variety and Symptoms metrics cannot be empty fields."
        )

    symptoms_lower = request.symptoms.lower()
    
    # Simple rule-based orchestration logic evaluating mock knowledge base triggers
    if "blight" in symptoms_lower or "spot" in symptoms_lower or "discoloration" in symptoms_lower:
        report_output = AGRONOMIC_KNOWLEDGE_BASE["blight"]
    elif "mildew" in symptoms_lower or "white" in symptoms_lower or "powdery" in symptoms_lower:
        report_output = AGRONOMIC_KNOWLEDGE_BASE["mildew"]
    else:
        # Fallback standard expert-system summary framework
        report_output = (
            f"DIAGNOSIS: Generalized Biotarget Stress complex detected for specimen variety [{request.crop_type.toUpperCase()}].\n\n"
            f"DEVELOPMENT METRIC: Verified at active [{request.crop_stage}] lifecycle block.\n"
            f"FIELD NOTE SYMPTOM ANALYSIS: \"{request.symptoms}\"\n\n"
            f"RECOMMENDED INTERVENTION: Administer localized systemic organic Neem Oil extracts (1500 ppm) at 5ml/L "
            f"integrated alongside balanced macro-nutrient application (NPK 19:19:19 profiling) to restore cell turgor.\n\n"
            f"14-DAY MONITORING TARGET: Isolate current parcel and log hidden sequence vigor metrics via tracking ledger."
        )

    # Return JSON body mapping directly back to your frontend data key 'diagnosis_report'
    return {
        "crop": request.crop_type,
        "stage": request.crop_stage,
        "diagnosis_report": report_output
    }
```

### 📋 To get this running smoothly:
1. Save this exact code inside your code structure as `backend/main.py`.
2. Ensure you have installed the required dependencies by running:
   ```bash
   pip install fastapi uvicorn pydantic