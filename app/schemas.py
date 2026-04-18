from __future__ import annotations 
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    sqft: int = Field(alias="SqFt", ge=0, le=10000000)
    bedrooms: int = Field(alias="Bedrooms", ge=0, le=100)
    bathrooms: int = Field(alias="Bathrooms", ge=0, le=50)
    offers: int = Field(alias="Offers", ge=0, le=1000)
    brick: str = Field(alias="Brick", min_length=2, max_length=3)
    neighborhood: str = Field(alias="Neighborhood", min_length=4, max_length=5)
    
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "SqFt": 10000,
                "Bedrooms": 50,
                "Bathrooms": 40,
                "Offers": 5,
                "Brick": "Yes",
                "Neighborhood": "ABCD"
            }
        }
    }

class PredictionResponse(BaseModel):
    predicted_price: float