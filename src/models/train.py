"""
Model Training & MLflow Experiment Tracking Module.
Trains Ridge, XGBoost, LightGBM, and CatBoost models with Time-Based Cross Validation.
Logs all hyperparameters, cross-validation metrics, and artifacts to MLflow.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

from src.utils.logger import get_logger

logger = get_logger("train")

def compute_metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100.0)
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MedAE": round(medae, 4),
        "MAPE": round(mape, 4)
    }

def train_and_evaluate(params_path: str = "params.yaml"):
    with open(params_path, 'r') as f:
        config = yaml.safe_load(f)
        
    train_cfg = config['train']
    input_path = train_cfg['input_path']
    model_output_path = train_cfg['model_output_path']
    metrics_output_path = train_cfg['metrics_output_path']
    target_col = train_cfg['target_column']
    n_splits = train_cfg.get('n_splits', 5)
    
    logger.info(f"Loading engineered dataset from {input_path}")
    df = pd.read_csv(input_path) if input_path.endswith('.csv') else pd.read_parquet(input_path)
    
    # Sort chronologically for time-based split
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Feature columns selection
    drop_cols = ['load_id', 'date', 'pickup', 'delivery', 'lane', 'equipment', target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    
    X = df[feature_cols].copy()
    y = df[target_col].values
    
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}. Features count: {len(feature_cols)}")
    
    # Setup MLflow
    mlflow.set_experiment("Spotter_Freight_Rate_Prediction")
    
    models_to_run = train_cfg.get('models_to_train', ['ridge', 'xgboost', 'lightgbm', 'catboost'])
    model_candidates = {}
    
    if 'ridge' in models_to_run:
        model_candidates['Ridge'] = Ridge(alpha=10.0, random_state=42)
        
    if 'xgboost' in models_to_run and xgb is not None:
        xgb_params = train_cfg.get('xgboost', {})
        model_candidates['XGBoost'] = xgb.XGBRegressor(
            n_estimators=xgb_params.get('n_estimators', 500),
            learning_rate=xgb_params.get('learning_rate', 0.05),
            max_depth=xgb_params.get('max_depth', 6),
            subsample=xgb_params.get('subsample', 0.8),
            colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1
        )
        
    if 'lightgbm' in models_to_run and lgb is not None:
        lgb_params = train_cfg.get('lightgbm', {})
        model_candidates['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=lgb_params.get('n_estimators', 500),
            learning_rate=lgb_params.get('learning_rate', 0.05),
            num_leaves=lgb_params.get('num_leaves', 31),
            subsample=lgb_params.get('subsample', 0.8),
            colsample_bytree=lgb_params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
    if 'catboost' in models_to_run and cb is not None:
        cb_params = train_cfg.get('catboost', {})
        model_candidates['CatBoost'] = cb.CatBoostRegressor(
            iterations=cb_params.get('iterations', 500),
            learning_rate=cb_params.get('learning_rate', 0.05),
            depth=cb_params.get('depth', 6),
            verbose=0,
            random_seed=42
        )
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_overall_model = None
    best_overall_mae = float('inf')
    best_model_name = ""
    summary_results = {}

    for name, model in model_candidates.items():
        logger.info(f"--- Training & Evaluating Model: {name} ---")
        
        cv_metrics_list = []
        oof_preds = np.zeros(len(df))
        
        with mlflow.start_run(run_name=f"Model_{name}"):
            # Log params
            mlflow.log_param("model_name", name)
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("n_features", len(feature_cols))
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, y_train = X.iloc[train_idx], y[train_idx]
                X_val, y_val = X.iloc[val_idx], y[val_idx]
                
                # Fit fold
                if name == 'CatBoost':
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
                elif name == 'XGBoost':
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                elif name == 'LightGBM':
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    model.fit(X_train, y_train)
                    
                preds = model.predict(X_val)
                oof_preds[val_idx] = preds
                
                fold_m = compute_metrics(y_val, preds)
                cv_metrics_list.append(fold_m)
                
            # Aggregate CV metrics over validation splits
            # Only compute overall metrics over filled OOF validation indices
            val_indices = np.concatenate([v for _, v in tscv.split(X)])
            overall_metrics = compute_metrics(y[val_indices], oof_preds[val_indices])
            
            logger.info(f"[{name}] Time-Series CV Overall Results: MAE=${overall_metrics['MAE']}, RMSE=${overall_metrics['RMSE']}, R2={overall_metrics['R2']}")
            
            # Log metrics to MLflow
            for m_key, m_val in overall_metrics.items():
                mlflow.log_metric(f"cv_{m_key}", m_val)
                
            # Fit model on all data for final export
            model.fit(X, y)
            mlflow.log_metric("final_train_mae", float(mean_absolute_error(y, model.predict(X))))
            
            summary_results[name] = overall_metrics
            
            if overall_metrics['MAE'] < best_overall_mae:
                best_overall_mae = overall_metrics['MAE']
                best_overall_model = model
                best_model_name = name
                
    logger.info(f"🏆 BEST MODEL: {best_model_name} with Validation MAE = ${best_overall_mae}")
    
    # Save best model artifact
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({"model": best_overall_model, "feature_names": feature_cols, "model_name": best_model_name}, model_output_path)
    
    metrics_payload = {
        "best_model": best_model_name,
        "best_validation_mae": best_overall_mae,
        "metrics_by_model": summary_results,
        "feature_count": len(feature_cols)
    }
    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
        
    logger.info(f"Best model saved to {model_output_path}")
    logger.info(f"Training metrics logged to {metrics_output_path}")

if __name__ == "__main__":
    train_and_evaluate()
