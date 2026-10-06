"""
Exploratory Data Analysis (EDA) Script for Spotter Freight Rate Prediction.
Generates comprehensive analysis of training, validation, and december input datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def run_eda(data_path: str = "data/raw/train-test.csv", val_path: str = "data/raw/validation.csv"):
    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    train = pd.read_csv(data_path)
    val = pd.read_csv(val_path)
    
    print(f"\n[1] DATASET SHAPES")
    print(f"  - Train/Test dataset: {train.shape[0]:,} rows, {train.shape[1]} columns")
    print(f"  - Validation dataset: {val.shape[0]:,} rows, {val.shape[1]} columns")
    
    print(f"\n[2] TRAIN DATA TYPES & NULL COUNTS")
    null_df = pd.DataFrame({
        'Dtype': train.dtypes,
        'Null_Count': train.isnull().sum(),
        'Null_Pct': (train.isnull().sum() / len(train) * 100).round(2),
        'N_Unique': train.nunique()
    })
    print(null_df.to_string())
    
    print(f"\n[3] DATA QUALITY ISSUES DISCOVERED")
    neg_weights = (train['weight'] < 0).sum()
    print(f"  - Negative weight values: {neg_weights} rows (Min: {train['weight'].min()}, Max: {train['weight'].max()})")
    print(f"  - Missing weight values: {train['weight'].isnull().sum()} rows")
    print(f"  - Missing market_index values: {train['market_index'].isnull().sum()} rows")
    
    print(f"\n[4] TARGET VARIABLE ('posted_rate') ANALYSIS")
    print(train['posted_rate'].describe().round(2).to_string())
    
    print(f"\n[5] NUMERICAL FEATURE CORRELATIONS WITH TARGET")
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    corrs = train[numeric_cols].corr()['posted_rate'].sort_values(ascending=False)
    print(corrs.to_string())
    
    print(f"\n[6] TEMPORAL DISTRIBUTIONS")
    train['date'] = pd.to_datetime(train['date'])
    val['date'] = pd.to_datetime(val['date'])
    print(f"  - Train Date Range: {train['date'].min().strftime('%Y-%m-%d')} to {train['date'].max().strftime('%Y-%m-%d')} ({train['date'].dt.month.nunique()} months)")
    print(f"  - Validation Date Range: {val['date'].min().strftime('%Y-%m-%d')} to {val['date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n[7] EQUIPMENT BREAKDOWN")
    print("Train Equipment Counts:")
    print(train['equipment'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
    print("\n" + "=" * 70)
    print("EDA COMPLETE.")
    print("=" * 70)

if __name__ == "__main__":
    run_eda()
