"""
Unit tests for data pipeline loading and validation.
"""

import pandas as pd
import pytest
from utils.feature_store import feature_store


def test_load_feature_store():
    df = feature_store.load_features()
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "city" in df.columns
        cities = df["city"].unique()
        assert len(cities) > 0


def test_feature_store_telemetry():
    status = feature_store.get_status()
    assert isinstance(status, dict)
    assert "active_store_type" in status
