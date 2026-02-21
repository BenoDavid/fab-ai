# src/app.py
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from src.schemas import PredictRequest, PredictBatchRequest, PredictResponse, HealthResponse
from src.utils import load_artifacts, build_dataframe_from_payload

MODEL_DIR = os.environ.get("MODEL_DIR", "model")

app = FastAPI(
    title="Needed Quantity Predictor",
    description="Predicts 'needed_qty' for fabrics given production inputs. Supports partial inputs.",
    version="1.0.0"
)

# Load model on startup
try:
    model, meta = load_artifacts(MODEL_DIR)
except Exception as e:
    model, meta = None, None
    load_error = str(e)
else:
    load_error = None

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if model is not None else f"error: {load_error}",
                          model_loaded=model is not None)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model, meta
    if model is None or meta is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    feature_cols = meta["feature_cols"]
    categorical_cols = meta["categorical_cols"]
    numeric_cols = meta["numeric_cols"]

    payload = req.model_dump(exclude_none=True)
    df = build_dataframe_from_payload(payload, feature_cols, categorical_cols, numeric_cols)

    # Track missing features for transparency
    missing_features = [c for c in feature_cols if c not in payload]

    try:
        pred = model.predict(df)[0]
        return PredictResponse(
            needed_qty=float(pred),
            features_used=[c for c in feature_cols if c in payload],
            missing_features=missing_features
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/predict-batch")
def predict_batch(req: PredictBatchRequest):
    global model, meta
    if model is None or meta is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    feature_cols = meta["feature_cols"]
    categorical_cols = meta["categorical_cols"]
    numeric_cols = meta["numeric_cols"]

    frames = []
    missing_info = []
    for item in req.items:
        payload = item.model_dump(exclude_none=True)
        df = build_dataframe_from_payload(payload, feature_cols, categorical_cols, numeric_cols)
        frames.append(df)
        missing_info.append([c for c in feature_cols if c not in payload])

    X = pd.concat(frames, axis=0, ignore_index=True)
    try:
        preds = model.predict(X)
        out = []
        for pred, missing in zip(preds, missing_info):
            out.append({
                "needed_qty": float(pred),
                "missing_features": missing
            })
        return JSONResponse(content={"predictions": out})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")