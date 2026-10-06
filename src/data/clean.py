"""
Data Cleaning Module for Spotter Freight Rate Prediction.
Handles data quality issues: negative weights, missing values, date parsing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from src.utils.logger import get_logger

logger = get_logger("data_clean")

def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean the input dataframe.
    - Takes absolute value of negative weights.
    - Imputes missing weights based on median per equipment type.
    - Imputes missing market_index using forward-fill / back-fill by date, then median fallback.
    """
    cleaned = df.copy()
    
    # 1. Parse date
    cleaned['date'] = pd.to_datetime(cleaned['date'])
    
    # 2. Fix negative weights (take abs value as negative weight is physically impossible)
    if 'weight' in cleaned.columns:
        neg_count = (cleaned['weight'] < 0).sum()
        if neg_count > 0:
            logger.info(f"Converting {neg_count} negative weight values to positive (abs).")
            cleaned['weight'] = cleaned['weight'].abs()
            
        # Impute missing weight per equipment type
        if cleaned['weight'].isnull().any():
            missing_count = cleaned['weight'].isnull().sum()
            logger.info(f"Imputing {missing_count} missing weight values by equipment median.")
            equipment_medians = cleaned.groupby('equipment')['weight'].transform('median')
            cleaned['weight'] = cleaned['weight'].fillna(equipment_medians)
            # Fallback if any still null
            cleaned['weight'] = cleaned['weight'].fillna(32000.0)
            
    # 3. Impute missing market_index
    if 'market_index' in cleaned.columns:
        if cleaned['market_index'].isnull().any():
            missing_count = cleaned['market_index'].isnull().sum()
            logger.info(f"Imputing {missing_count} missing market_index values via temporal forward-fill.")
            cleaned = cleaned.sort_values('date')
            cleaned['market_index'] = cleaned['market_index'].ffill().bfill()
            # Fallback to global median
            cleaned['market_index'] = cleaned['market_index'].fillna(cleaned['market_index'].median())
            
    return cleaned

def run_cleaning_pipeline(params_path: str = "params.yaml"):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)['prepare']
        
    raw_path = params['raw_train_path']
    output_path = params['processed_train_path']
    
    logger.info(f"Loading raw dataset from {raw_path}")
    raw_df = pd.read_csv(raw_path)
    
    cleaned_df = clean_data(raw_df, is_train=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved successfully to {output_path} ({len(cleaned_df):,} rows)")

if __name__ == "__main__":
    run_cleaning_pipeline()
