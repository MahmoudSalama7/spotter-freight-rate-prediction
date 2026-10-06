"""
Unit tests for Data Cleaning module.
"""

import pandas as pd
import pytest
from src.data.clean import clean_data

def test_clean_data_handles_negatives_and_nulls():
    raw_df = pd.DataFrame({
        'date': ['2025-01-01', '2025-01-02'],
        'equipment': ['Dry Van', 'Reefer'],
        'weight': [-30000.0, None],
        'market_index': [1.05, None]
    })
    
    cleaned = clean_data(raw_df, is_train=True)
    
    assert (cleaned['weight'] >= 0).all(), "Negative weights should be converted to positive"
    assert not cleaned['weight'].isnull().any(), "Missing weights should be imputed"
    assert not cleaned['market_index'].isnull().any(), "Missing market_index should be imputed"
