from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PredictRequest, PredictResponse
from app.services.inference_service import run_inference

router = APIRouter()

AVAILABLE_MODELS = [
    {
        "id": "distilgpt2",
        "name": "DistilGPT2",
        "status": "available"
    },
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen2.5-0.5B",
        "status": "available"
    },
    {
        "id": "tinyllama",
        "name": "TinyLlama-1.1B",
        "status": "available"
    }
]

@router.get("/")
def get_status():
    """Health check endpoint"""
    return {"status": "ok", "message": "SLM-QA-Comparison Backend is running"}

@router.get("/models")
def get_models():
    """Return the list of available models"""
    return {"models": AVAILABLE_MODELS}

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Run Q&A inference using the selected model"""
    try:
        return run_inference(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
