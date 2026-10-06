"""
FastAPI Model Serving Application for Spotter Freight Rate Prediction.
"""

from fastapi import FastAPI, HTTPException, status
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from api.schemas import LoadInput, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from src.data.clean import clean_data
from src.data.features import FeaturePipeline

# Global models and pipeline storage
ML_MODELS: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and pipeline on startup
    pipeline_path = Path("models/feature_pipeline.joblib")
    model_path = Path("models/best_model.joblib")
    
    if not pipeline_path.exists() or not model_path.exists():
        print("[WARNING] Pipeline or Model artifact not found. Please run training pipeline first.")
    else:
        ML_MODELS["pipeline"] = joblib.load(pipeline_path)
        artifacts = joblib.load(model_path)
        ML_MODELS["model"] = artifacts["model"]
        ML_MODELS["feature_names"] = artifacts["feature_names"]
        ML_MODELS["model_name"] = artifacts["model_name"]
        print(f"[INFO] Loaded model {artifacts['model_name']} and feature pipeline successfully.")
    yield
    ML_MODELS.clear()

app = FastAPI(
    title="Spotter Freight Rate Prediction API",
    description="MLOps Serving Endpoint for predicting truckload freight rates ($)",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    model_loaded = "model" in ML_MODELS and "pipeline" in ML_MODELS
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_name": ML_MODELS.get("model_name", "N/A")
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(load: LoadInput):
    if "model" not in ML_MODELS:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
        
    try:
        # Convert Pydantic object to dataframe
        input_dict = load.model_dump()
        df = pd.DataFrame([input_dict])
        
        # Preprocess & feature transform
        cleaned_df = clean_data(df, is_train=False)
        featured_df = ML_MODELS["pipeline"].transform(cleaned_df)
        
        X = featured_df[ML_MODELS["feature_names"]]
        pred = float(ML_MODELS["model"].predict(X)[0])
        pred = max(10.0, round(pred, 2))
        
        return PredictionResponse(
            load_id=load.load_id or "TE-000000",
            predicted_rate=pred
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchPredictionRequest):
    if "model" not in ML_MODELS:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
        
    try:
        input_dicts = [load.model_dump() for load in batch.loads]
        df = pd.DataFrame(input_dicts)
        
        cleaned_df = clean_data(df, is_train=False)
        featured_df = ML_MODELS["pipeline"].transform(cleaned_df)
        
        X = featured_df[ML_MODELS["feature_names"]]
        preds = ML_MODELS["model"].predict(X)
        preds = np.maximum(10.0, np.round(preds, 2))
        
        responses = [
            PredictionResponse(load_id=load.load_id or f"TE-{i:06d}", predicted_rate=float(pred))
            for i, (load, pred) in enumerate(zip(batch.loads, preds), 1)
        ]
        
        return BatchPredictionResponse(
            predictions=responses,
            total_count=len(responses)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
