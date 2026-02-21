# src/schemas.py
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    # All fields optional to allow "any one input"
    po: Optional[str] = Field(None, description="Purchase order")
    style: Optional[str] = Field(None, description="Style code/name")
    total_qty_to_produce: Optional[float] = Field(None, description="Total qty to produce")
    estimated_fabrics_needed: Optional[float] = Field(None, description="Estimated fabric needed")
    requested_fabrics_qty: Optional[float] = Field(None, description="Requested fabric qty")

class PredictBatchRequest(BaseModel):
    items: List[PredictRequest]

class PredictResponse(BaseModel):
    needed_qty: float
    features_used: List[str]
    missing_features: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool