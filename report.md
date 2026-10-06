# Freight Rate Prediction — Final Technical Report

## 1. Executive Summary
This project addresses the Spotter Freight Rate Prediction challenge. Using end-to-end MLOps practices, we developed a production-ready freight pricing solution that predicts spot market rates (`posted_rate`) based on lane distance, equipment type, load weight, market index signals, quote signals, and temporal characteristics.

- **Primary Model**: CatBoost Regressor
- **Validation MAE**: **$133.49**
- **Validation R²**: **0.8365**
- **Validation RMSE**: **$607.06**
- **Validation MedAE**: **$58.18**
- **Validation MAPE**: **7.09%**

---

## 2. Exploratory Data Analysis & Data Quality Findings

| Metric / Aspect | Value / Observation | Action Taken |
|---|---|---|
| **Training Sample Size** | 48,000 loads (Jan 1, 2025 – Oct 31, 2025) | Used for Time-Series CV training |
| **Validation Sample Size** | 12,000 loads (Nov 1, 2025 – Nov 30, 2025) | Prediction target set (`validation_predictions.csv`) |
| **December Forecast Target** | 31 loads (Dec 1, 2025 – Dec 31, 2025) | Fixed Lexington $\to$ Fort Wayne lane forecast |
| **Negative Weights** | 292 rows in training set (e.g. -47,500 lbs) | Converted to absolute values (`df['weight'].abs()`) |
| **Missing Weights** | 300 missing values | Imputed via equipment-type median values |
| **Missing Market Index** | 374 missing values | Temporal forward-fill ($\text{ffill} \to \text{bfill}$) |

---

## 3. Data Split & Validation Strategy

> [!IMPORTANT]
> **Temporal Cross-Validation Strategy**
> Since freight rates depend heavily on seasonal trends and market dynamics over time, a random K-Fold split would introduce target data leakage from future loads into past training folds.
> 
> We implemented a **5-Split TimeSeriesSplit (Expanding Window)**:
> - Fold 1: Train Jan–Feb $\to$ Validate Mar
> - Fold 2: Train Jan–Mar $\to$ Validate Apr
> - Fold 3: Train Jan–Apr $\to$ Validate May
> - ...
> - Fold 5: Train Jan–Sep $\to$ Validate Oct

---

## 4. Feature Engineering

1. **Spatial & Geo Features**:
   - Haversine distance calculated from latitude/longitude coordinates.
   - $\Delta \text{Latitude}$ and $\Delta \text{Longitude}$ absolute differences.
   - Ratio of posted distance to Haversine distance ($\frac{\text{distance}}{\text{haversine\_dist}}$).
   - Dynamic lookup mapping of city coordinates for unseen prediction inputs.

2. **Temporal Features**:
   - Cyclical sin/cos encodings for `day_of_year` and `day_of_week`.
   - `is_weekend` indicator and month features.

3. **Domain Interaction Features**:
   - `weight_per_mile` ($\frac{\text{weight}}{\text{distance} + 1}$)
   - `dist_x_market` ($\text{distance} \times \text{market\_index}$)
   - `dist_x_quote` ($\text{distance} \times \text{quote\_signal}$)
   - Log transformations on skewed numeric variables (`log_distance`, `log_weight`).

---

## 5. Model Benchmarking & MLflow Results

All experiments were tracked using **MLflow** with time-based cross-validation metrics:

| Model Candidate | Validation MAE ($) | Validation RMSE ($) | R² Score | MedAE ($) | MAPE (%) |
|---|---|---|---|---|---|
| **CatBoost Regressor (Selected)** | **$133.49** | **$607.06** | **0.8365** | **$58.18** | **7.09%** |
| Ridge Regression (Baseline) | $157.41 | $617.19 | 0.8310 | $76.40 | 8.37% |
| LightGBM Regressor | $179.57 | $640.08 | 0.8182 | $74.52 | 8.44% |
| XGBoost Regressor | $242.52 | $718.55 | 0.7709 | $103.20 | 11.77% |

---

## 6. December Fixed Lane Forecast Analysis

For December 2025, inputs are fixed for the Lexington to Fort Wayne route (360 miles, Dry Van, 32,000 lbs).

The CatBoost model predicts rate fluctuations across December based on day-of-week seasonality (higher rate demand before weekends and holiday periods).

Validation via `score.py` passed with 0 errors, producing `scorer_results/candidate_december.png`.

---

## 7. MLOps System Architecture & Reproducibility

- **Data Versioning (DVC)**: Managed raw datasets, cleaning, featurization, and prediction stages (`dvc.yaml`).
- **Experiment Tracking (MLflow)**: Recorded parameters, time-series metrics, and model registry artifacts.
- **Serving (FastAPI)**: RESTful API supporting single (`/predict`) and batch (`/predict/batch`) inference with health check monitoring.
- **Containerization (Docker)**: Multi-stage container setup (`Dockerfile` & `docker-compose.yml`) running the API service and MLflow UI.
- **Testing**: Automated `pytest` suite validating data cleaning, feature transformations, and API health endpoints.
