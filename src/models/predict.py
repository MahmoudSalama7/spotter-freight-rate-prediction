"""
Inference Module for Validation and December Forecast Predictions.
Applies data cleaning and feature engineering pipelines, loads trained model,
and outputs validation_predictions.csv and data/december_chart_inputs.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import yaml

from src.data.clean import clean_data
from src.data.features import FeaturePipeline
from src.utils.logger import get_logger

logger = get_logger("predict")

def generate_predictions(params_path: str = "params.yaml"):
    with open(params_path, 'r') as f:
        config = yaml.safe_load(f)
        
    pred_cfg = config['predict']
    features_cfg = config['features']
    model_cfg = config['train']
    
    val_input = pred_cfg['validation_input']
    dec_input = pred_cfg['december_input']
    val_output = pred_cfg['validation_output']
    dec_output = pred_cfg['december_output']
    
    pipeline_path = features_cfg['pipeline_path']
    model_path = model_cfg['model_output_path']
    
    logger.info(f"Loading feature pipeline from {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    
    logger.info(f"Loading trained best model from {model_path}")
    model_artifacts = joblib.load(model_path)
    model = model_artifacts['model']
    feature_names = model_artifacts['feature_names']
    
    # 1. Predict Validation Set (12,000 rows)
    logger.info(f"Processing validation set from {val_input}")
    val_df = pd.read_csv(val_input)
    val_cleaned = clean_data(val_df, is_train=False)
    val_featured = pipeline.transform(val_cleaned)
    
    X_val = val_featured[feature_names]
    val_preds = model.predict(X_val)
    # Ensure positive rates
    val_preds = np.maximum(val_preds, 10.0)
    
    val_results = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": np.round(val_preds, 2)
    })
    
    val_results.to_csv(val_output, index=False)
    logger.info(f"Successfully generated validation predictions -> {val_output} ({len(val_results):,} rows)")
    
    # 2. Predict December Set (31 rows)
    logger.info(f"Processing December inputs from {dec_input}")
    dec_df = pd.read_csv(dec_input)
    dec_cleaned = clean_data(dec_df, is_train=False)
    dec_featured = pipeline.transform(dec_cleaned)
    
    X_dec = dec_featured[feature_names]
    dec_preds = model.predict(X_dec)
    dec_preds = np.maximum(dec_preds, 10.0)
    
    dec_df["predicted_rate"] = np.round(dec_preds, 2)
    
    # Save to data/december_chart_inputs.csv and dec_output
    Path(dec_output).parent.mkdir(parents=True, exist_ok=True)
    dec_df.to_csv(dec_output, index=False)
    dec_df.to_csv("december-chart-inputs.csv", index=False)
    
    logger.info(f"Successfully generated December predictions -> {dec_output} (31 rows)")

if __name__ == "__main__":
    generate_predictions()
