# 🚚 Spotter — Freight Rate Prediction

> **End-to-end MLOps solution** for predicting U.S. truckload freight rates, built with CatBoost, DVC pipelines, MLflow experiment tracking, FastAPI serving, Docker containerization, and automated testing.

| Metric | Value |
|--------|-------|
| **Best Model** | CatBoost |
| **Validation MAE** | $133.49 |
| **R²** | 0.8365 |
| **MAPE** | 7.09% |
| **Features** | 33 engineered |
| **Training Samples** | 48,000 loads |

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Model Benchmarks](#model-benchmarks)
- [Feature Engineering](#feature-engineering)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [December Forecast](#december-forecast)

---

## Overview

This project predicts **posted freight rates ($)** for truckloads across the United States. Given features like pickup/delivery cities, distance, equipment type, weight, and market signals, the model outputs a dollar-value rate prediction.

### Key Design Decisions

1. **CatBoost** selected over Ridge, XGBoost, and LightGBM — lowest MAE ($133) and highest R² (0.84). CatBoost's ordered boosting prevents overfitting on time-ordered data.
2. **TimeSeriesSplit (5 folds)** instead of random K-Fold — freight rates are time-dependent, so the model always trains on the past and validates on the future.
3. **DVC pipelines** for full reproducibility — every stage from cleaning to prediction is tracked and can be re-run with `dvc repro`.
4. **MLflow** for experiment tracking — all hyperparameters, cross-validation metrics, and model artifacts are logged.

---

## Project Structure

```
Spotter/
├── api/                          # FastAPI inference service
│   ├── main.py                   # App with /predict, /predict/batch, /health
│   └── schemas.py                # Pydantic request/response models
├── data/
│   ├── raw/                      # Raw input CSVs (git-ignored)
│   └── processed/                # Cleaned & featured CSVs (git-ignored)
├── models/                       # Saved model & feature pipeline (git-ignored)
│   ├── best_model.joblib
│   └── feature_pipeline.joblib
├── metrics/
│   └── train_metrics.json        # Cross-validation results
├── notebooks/
│   └── eda_summary.py            # Exploratory data analysis
├── scorer_results/               # Validation outputs & charts
│   ├── candidate_december.png
│   └── plots/
├── src/
│   ├── data/
│   │   ├── clean.py              # Data cleaning (negative weights, imputation)
│   │   └── features.py           # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py              # Model training & MLflow tracking
│   │   ├── evaluate.py           # Model evaluation
│   │   └── predict.py            # Batch prediction generation
│   └── utils/
│       └── logger.py             # Logging utility
├── tests/
│   ├── test_api.py               # API endpoint tests
│   ├── test_data.py              # Data cleaning tests
│   └── test_features.py          # Feature pipeline tests
├── conftest.py                   # Pytest configuration
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # Hyperparameters & paths config
├── score.py                      # Official scorer (validates outputs)
├── Dockerfile                    # Container for API serving
├── docker-compose.yml            # API + MLflow UI services
├── Makefile                      # Convenience commands
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/spotter-freight-rate-prediction.git
cd spotter-freight-rate-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place the raw training CSV in the expected location:

```bash
mkdir -p data/raw
# Copy train-test.csv → data/raw/train-test.csv
# Copy validation.csv → data/raw/validation.csv
# Copy december-chart-inputs.csv → data/raw/december-chart-inputs.csv
```

### 3. Run the Full Pipeline

**Option A — DVC (recommended for reproducibility):**

```bash
dvc repro
```

This runs all five stages sequentially: `clean → features → train → evaluate → predict`.

**Option B — Makefile (stage by stage):**

```bash
make clean        # Clean & impute raw data
make features     # Engineer 33 features
make train        # Train & benchmark 4 models
make evaluate     # Generate evaluation plots
make predict      # Produce validation_predictions.csv
```

**Option C — Individual Python modules:**

```bash
python -m src.data.clean
python -m src.data.features
python -m src.models.train
python -m src.models.evaluate
python -m src.models.predict
```

### 4. Validate Outputs

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

Expected output:
```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

---

## Pipeline Stages

The pipeline is defined in [`dvc.yaml`](dvc.yaml) with five reproducible stages:

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| **clean** | `src.data.clean` | `data/raw/train-test.csv` | `data/processed/train_cleaned.csv` |
| **features** | `src.data.features` | Cleaned CSV | `data/processed/train_features.csv` + `models/feature_pipeline.joblib` |
| **train** | `src.models.train` | Featured CSV | `models/best_model.joblib` + `metrics/train_metrics.json` |
| **evaluate** | `src.models.evaluate` | Model + featured CSV | Evaluation plots |
| **predict** | `src.models.predict` | Model + pipeline + validation CSV | `validation_predictions.csv` + December forecast |

All hyperparameters are centralized in [`params.yaml`](params.yaml).

---

## Model Benchmarks

Four model families were benchmarked using **TimeSeriesSplit (5 expanding-window folds)**, all tracked in MLflow:

| Model | MAE ($) | RMSE ($) | R² | MedAE ($) | MAPE (%) |
|-------|---------|----------|------|-----------|----------|
| **CatBoost** ✅ | **133.49** | **607.06** | **0.8365** | **58.18** | **7.09** |
| Ridge | 157.41 | 617.19 | 0.8310 | 76.40 | 8.37 |
| LightGBM | 179.57 | 640.08 | 0.8182 | 74.52 | 8.44 |
| XGBoost | 242.52 | 718.55 | 0.7709 | 103.20 | 11.77 |

### Why CatBoost?

- **Lowest MAE** ($133) and **highest R²** (0.84)
- Handles categorical features natively (equipment type)
- **Ordered boosting** prevents target leakage on time-ordered data
- Built-in support for early stopping

---

## Feature Engineering

33 features are computed in [`src/data/features.py`](src/data/features.py):

### Data Cleaning ([`src/data/clean.py`](src/data/clean.py))
- **Negative weights** (292 rows): Converted to absolute values
- **Missing weights** (~300 rows): Imputed using median weight per equipment type
- **Missing market_index** (374 rows): Temporal forward-fill then back-fill

### Engineered Features

| Category | Features |
|----------|----------|
| **Temporal** | `day_of_week`, `day_of_month`, `month`, `day_of_year`, `is_weekend` |
| **Cyclical Encoding** | `sin_day_of_year`, `cos_day_of_year`, `sin_day_of_week`, `cos_day_of_week` |
| **Spatial** | `haversine_dist`, `delta_lat`, `delta_lon`, `dist_to_haversine_ratio` |
| **Equipment** | `equipment_code`, `is_dry_van`, `is_reefer`, `is_flatbed` |
| **Location** | `pickup_freq`, `delivery_freq` (frequency encoding) |
| **Domain Interactions** | `weight_per_mile`, `dist_x_market`, `dist_x_quote`, `market_x_quote` |
| **Transforms** | `log_distance`, `log_weight` |

---

## API Reference

The FastAPI application (`api/main.py`) serves predictions via REST endpoints.

### Start the API

```bash
# Local
uvicorn api.main:app --reload --port 8000

# Or via Makefile
make serve
```

### Endpoints

#### `GET /health`
Health check — returns model loading status.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "CatBoost"
}
```

#### `POST /predict`
Single load prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32000,
    "date": "2025-12-15"
  }'
```

```json
{
  "load_id": "TE-000000",
  "predicted_rate": 852.47,
  "unit": "USD",
  "model_version": "1.0.0"
}
```

#### `POST /predict/batch`
Batch prediction for multiple loads.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "loads": [
      {"pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360, "equipment": "Dry Van", "weight": 32000, "date": "2025-12-01"},
      {"pickup": "Chicago", "delivery": "Dallas", "distance": 920, "equipment": "Reefer", "weight": 44000, "date": "2025-12-01"}
    ]
  }'
```

#### Interactive Docs

Once the API is running, visit **http://localhost:8000/docs** for the Swagger UI.

---

## Docker Deployment

### Build & Run the API Container

```bash
# Build
docker build -t spotter-freight-api:latest .

# Run
docker run -p 8000:8000 spotter-freight-api:latest
```

### Docker Compose (API + MLflow UI)

```bash
docker-compose up -d
```

This starts:
- **API** on `http://localhost:8000`
- **MLflow UI** on `http://localhost:5000`

---

## Testing

Automated tests cover data cleaning, feature engineering, and API endpoints:

```bash
# Run all tests
pytest tests/

# Or via Makefile
make test
```

| Test File | Coverage |
|-----------|----------|
| `test_data.py` | Data cleaning logic, negative weight handling, null imputation |
| `test_features.py` | Feature pipeline fit/transform, feature count, column presence |
| `test_api.py` | API health check, prediction endpoints |

---

## December Forecast

The model produces a 31-day forecast for the **Lexington → Fort Wayne** Dry Van lane (360 miles, 32,000 lbs) across December 2025.

The chart shows learned **weekly cyclical patterns** — rates dip on weekends (Saturdays) and rise mid-week, with a slight upward trend approaching the holiday season.

Generate the chart:
```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

Output: `scorer_results/candidate_december.png`

---

## Configuration

All pipeline parameters are in [`params.yaml`](params.yaml):

```yaml
prepare:
  clip_rate_min: 50.0
  clip_rate_max: 26000.0

train:
  target_column: "posted_rate"
  n_splits: 5
  models_to_train: [ridge, xgboost, lightgbm, catboost]
  catboost:
    iterations: 500
    learning_rate: 0.05
    depth: 6
```

---

## License

This project was built as a solution for the Spotter Freight Rate Prediction challenge.