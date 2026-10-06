"""
Feature Engineering Pipeline Module.
Extracts temporal, spatial, domain interaction, and categorical features.
Supports fit/transform pattern to avoid data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logger import get_logger

logger = get_logger("features")

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth in miles.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 3956.0 * c
    return miles

class FeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pickup_freq_ = {}
        self.delivery_freq_ = {}
        self.equipment_map_ = {'Dry Van': 0, 'Reefer': 1, 'Flatbed': 2}
        self.global_market_median_ = 1.0
        self.global_quote_median_ = 2.0
        self.pickup_lat_map_ = {}
        self.pickup_lon_map_ = {}
        self.delivery_lat_map_ = {}
        self.delivery_lon_map_ = {}
        self.feature_columns_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        # Learn frequency encoding for high-cardinality locations
        self.pickup_freq_ = df['pickup'].value_counts(normalize=True).to_dict()
        self.delivery_freq_ = df['delivery'].value_counts(normalize=True).to_dict()
        
        # Learn city geo-coordinate mappings
        if 'pickup_lat' in df.columns:
            self.pickup_lat_map_ = df.groupby('pickup')['pickup_lat'].mean().to_dict()
            self.pickup_lon_map_ = df.groupby('pickup')['pickup_lon'].mean().to_dict()
        if 'delivery_lat' in df.columns:
            self.delivery_lat_map_ = df.groupby('delivery')['delivery_lat'].mean().to_dict()
            self.delivery_lon_map_ = df.groupby('delivery')['delivery_lon'].mean().to_dict()
            
        if 'market_index' in df.columns:
            self.global_market_median_ = df['market_index'].median()
        if 'quote_signal' in df.columns:
            self.global_quote_median_ = df['quote_signal'].median()
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # Ensure lat/lon columns are present by looking up city mappings if missing
        if 'pickup_lat' not in df.columns:
            df['pickup_lat'] = df['pickup'].map(self.pickup_lat_map_).fillna(35.6)
        if 'pickup_lon' not in df.columns:
            df['pickup_lon'] = df['pickup'].map(self.pickup_lon_map_).fillna(-90.9)
        if 'delivery_lat' not in df.columns:
            df['delivery_lat'] = df['delivery'].map(self.delivery_lat_map_).fillna(35.6)
        if 'delivery_lon' not in df.columns:
            df['delivery_lon'] = df['delivery'].map(self.delivery_lon_map_).fillna(-90.8)

        # 1. Date / Temporal Features
        date_series = pd.to_datetime(df['date'])
        df['day_of_week'] = date_series.dt.dayofweek
        df['day_of_month'] = date_series.dt.day
        df['month'] = date_series.dt.month
        df['day_of_year'] = date_series.dt.dayofyear
        df['is_weekend'] = (date_series.dt.dayofweek >= 5).astype(int)
        
        # Cyclical temporal encoding
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        
        # 2. Equipment Encoding
        df['equipment_code'] = df['equipment'].map(self.equipment_map_).fillna(0).astype(int)
        
        # One-hot equipment
        for eq in ['Dry Van', 'Reefer', 'Flatbed']:
            df[f'is_{eq.lower().replace(" ", "_")}'] = (df['equipment'] == eq).astype(int)
            
        # 3. Location Frequency & Pair Encoding
        df['pickup_freq'] = df['pickup'].map(self.pickup_freq_).fillna(0.0)
        df['delivery_freq'] = df['delivery'].map(self.delivery_freq_).fillna(0.0)
        df['lane'] = df['pickup'] + "_" + df['delivery']
        
        # 4. Spatial / Coordinates features
        df['haversine_dist'] = haversine_np(
            df['pickup_lon'], df['pickup_lat'],
            df['delivery_lon'], df['delivery_lat']
        )
        df['delta_lat'] = (df['delivery_lat'] - df['pickup_lat']).abs()
        df['delta_lon'] = (df['delivery_lon'] - df['pickup_lon']).abs()
        df['dist_to_haversine_ratio'] = df['distance'] / (df['haversine_dist'] + 1e-5)
            
        # 5. Missing values fill for market_index / quote_signal if missing
        if 'market_index' not in df.columns or df['market_index'].isnull().any():
            df['market_index'] = df.get('market_index', pd.Series(np.nan, index=df.index)).fillna(self.global_market_median_)
        if 'quote_signal' not in df.columns or df['quote_signal'].isnull().any():
            df['quote_signal'] = df.get('quote_signal', pd.Series(np.nan, index=df.index)).fillna(self.global_quote_median_)
            
        # 6. Domain Interaction Ratios
        df['weight_per_mile'] = df['weight'] / (df['distance'] + 1.0)
        df['dist_x_market'] = df['distance'] * df['market_index']
        df['dist_x_quote'] = df['distance'] * df['quote_signal']
        df['market_x_quote'] = df['market_index'] * df['quote_signal']
        df['log_distance'] = np.log1p(df['distance'])
        df['log_weight'] = np.log1p(np.maximum(0, df['weight']))
        
        # Select numeric features for model input
        feature_cols = [
            'distance', 'weight', 'market_index', 'quote_signal',
            'day_of_week', 'day_of_month', 'month', 'day_of_year', 'is_weekend',
            'sin_day_of_year', 'cos_day_of_year', 'sin_day_of_week', 'cos_day_of_week',
            'equipment_code', 'is_dry_van', 'is_reefer', 'is_flatbed',
            'pickup_freq', 'delivery_freq', 'haversine_dist', 'delta_lat', 'delta_lon',
            'dist_to_haversine_ratio', 'weight_per_mile', 'dist_x_market',
            'dist_x_quote', 'market_x_quote', 'log_distance', 'log_weight'
        ]
        
        self.feature_columns_ = feature_cols
        return df

def run_feature_pipeline(params_path: str = "params.yaml"):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)['features']
        
    input_path = params['input_path']
    output_path = params['output_path']
    pipeline_path = params['pipeline_path']
    
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info("Fitting and transforming feature pipeline...")
    pipeline = FeaturePipeline()
    featured_df = pipeline.fit_transform(df)
    
    # Save transformed features and pipeline
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pipeline_path).parent.mkdir(parents=True, exist_ok=True)
    
    featured_df.to_csv(output_path, index=False)
    joblib.dump(pipeline, pipeline_path)
    
    logger.info(f"Features saved to {output_path} ({featured_df.shape[1]} total columns).")
    logger.info(f"Feature pipeline saved to {pipeline_path}")

if __name__ == "__main__":
    run_feature_pipeline()
