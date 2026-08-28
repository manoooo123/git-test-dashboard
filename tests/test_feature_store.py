"""
Unit tests for Feature Store Manager (utils/feature_store.py).
"""

import pytest
import pandas as pd
from utils.feature_store import FeatureStoreManager, LOCAL_FEATURE_FILE


def test_feature_store_status():
    fs_mgr = FeatureStoreManager()
    status = fs_mgr.get_status()
    assert "hopsworks_configured" in status
    assert "active_store_type" in status
    assert "local_store_available" in status
    assert status["local_store_available"] is True


def test_feature_store_load():
    fs_mgr = FeatureStoreManager()
    df = fs_mgr.load_features()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "city" in df.columns
