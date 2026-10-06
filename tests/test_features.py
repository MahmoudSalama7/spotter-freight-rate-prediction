"""
Unit tests for Feature Pipeline module.
"""

import pandas as pd
import numpy as np
import pytest
from src.data.features import FeaturePipeline

def test_feature_pipeline_transformation():
    df = pd.DataFrame({
        'load_id': ['TE-000001'],
        'pickup': ['Lexington'],
        'delivery': ['Fort Wayne'],
        'pickup_lat': [38.0406],
        'pickup_lon': [-84.5037],
        'delivery_lat': [41.0793],
        'delivery_lon': [-85.1394],
        'distance': [360.0],
        'equipment': ['Dry Van'],
        'weight': [32000.0],
        'date': ['2025-12-01'],
        'market_index': [1.0],
        'quote_signal': [2.0]
    })
    
    pipeline = FeaturePipeline()
    featured = pipeline.fit_transform(df)
    
    assert 'weight_per_mile' in featured.columns
    assert 'haversine_dist' in featured.columns
    assert 'sin_day_of_year' in featured.columns
    assert featured['equipment_code'].iloc[0] == 0  # Dry Van -> 0
