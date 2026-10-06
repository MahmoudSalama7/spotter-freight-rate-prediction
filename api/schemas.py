"""
Pydantic Schemas for FastAPI Freight Rate Prediction Service.
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class LoadInput(BaseModel):
    load_id: Optional[str] = Field(default="TE-000000", description="Unique Load Identifier")
    pickup: str = Field(..., json_schema_extra={"example": "Lexington"})
    delivery: str = Field(..., json_schema_extra={"example": "Fort Wayne"})
    distance: float = Field(..., gt=0, json_schema_extra={"example": 360.0}, description="Distance in miles")
    equipment: str = Field(..., json_schema_extra={"example": "Dry Van"}, description="Dry Van, Reefer, or Flatbed")
    weight: float = Field(..., gt=0, json_schema_extra={"example": 32000.0}, description="Weight in lbs")
    date: str = Field(..., json_schema_extra={"example": "2025-12-01"}, description="Date string YYYY-MM-DD")
    pickup_lat: Optional[float] = Field(default=38.0406)
    pickup_lon: Optional[float] = Field(default=-84.5037)
    delivery_lat: Optional[float] = Field(default=41.0793)
    delivery_lon: Optional[float] = Field(default=-85.1394)
    market_index: Optional[float] = Field(default=1.0)
    quote_signal: Optional[float] = Field(default=2.0)

class PredictionResponse(BaseModel):
    load_id: str
    predicted_rate: float
    unit: str = "USD"
    model_version: str = "1.0.0"

class BatchPredictionRequest(BaseModel):
    loads: List[LoadInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
