```python
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Application Configuration
# ============================================================

app = FastAPI(
    title="Temperature Prediction API",
    description=(
        "REST API for predicting maximum and minimum temperatures "
        "using pre-trained regression models."
    ),
    version="1.0.0",
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


MAX_MODEL_PATH = MODEL_DIR / "max_temperature_model.pkl"
MIN_MODEL_PATH = MODEL_DIR / "min_temperature_model.pkl"


# ============================================================
# Load Pre-trained Models
# ============================================================

def load_model(model_path: Path):
    """
    Load a previously trained regression model.

    The model is loaded when the application starts.
    It is NOT trained again when a prediction request is received.
    """

    if not model_path.exists():
        return None

    return joblib.load(model_path)


max_model = load_model(MAX_MODEL_PATH)
min_model = load_model(MIN_MODEL_PATH)


# ============================================================
# Request Schema
# ============================================================

class PredictionRequest(BaseModel):
    num_days: int = Field(
        ...,
        gt=0,
        le=365,
        description="Number of future days to predict"
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Temperature Prediction API",
        "status": "running",
        "endpoints": [
            "POST /predict/max_t",
            "POST /predict/min_t"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "max_temperature_model_loaded": max_model is not None,
        "min_temperature_model_loaded": min_model is not None,
    }


# ============================================================
# Feature Preparation
# ============================================================

def create_future_features(num_days: int):
    """
    Create future dates and basic date-based features.

    IMPORTANT:
    The feature columns here must match the features used when
    training the models in web_scrapping.ipynb.
    """

    start_date = pd.Timestamp.today().normalize()

    dates = [
        start_date + timedelta(days=i)
        for i in range(1, num_days + 1)
    ]

    future_data = pd.DataFrame({
        "date": dates
    })

    # Basic calendar features
    future_data["year"] = future_data["date"].dt.year
    future_data["month"] = future_data["date"].dt.month
    future_data["day"] = future_data["date"].dt.day
    future_data["day_of_year"] = future_data["date"].dt.dayofyear

    return future_data


# ============================================================
# Prediction Helper
# ============================================================

def make_predictions(model, num_days: int):
    """
    Generate predictions using an already-trained model.
    """

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pre-trained model not found. "
                "Please train and save the model before starting the API."
            )
        )

    future_data = create_future_features(num_days)

    # Keep only numerical features.
    #
    # IMPORTANT:
    # These columns must be changed if the model in
    # web_scrapping.ipynb was trained using different features.
    feature_columns = [
        "year",
        "month",
        "day",
        "day_of_year"
    ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in future_data.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Missing feature columns: {missing_columns}"
        )

    X_future = future_data[feature_columns]

    predictions = model.predict(X_future)

    return future_data["date"], predictions


# ============================================================
# Maximum Temperature Prediction
# ============================================================

@app.post("/predict/max_t")
def predict_max_temperature(request: PredictionRequest):

    dates, predictions = make_predictions(
        max_model,
        request.num_days
    )

    result = []

    for date, prediction in zip(dates, predictions):
        result.append({
            "date": date.strftime("%Y-%m-%d"),
            "max_temperature": round(float(prediction), 2)
        })

    return {
        "predictions": result
    }


# ============================================================
# Minimum Temperature Prediction
# ============================================================

@app.post("/predict/min_t")
def predict_min_temperature(request: PredictionRequest):

    dates, predictions = make_predictions(
        min_model,
        request.num_days
    )

    result = []

    for date, prediction in zip(dates, predictions):
        result.append({
            "date": date.strftime("%Y-%m-%d"),
            "min_temperature": round(float(prediction), 2)
        })

    return {
        "predictions": result
    }
```
