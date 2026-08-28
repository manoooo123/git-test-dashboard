"""
Pearls AQI Predictor — End-to-End ML Pipeline Integration Test.

Validates the complete data flow:
  Feature store CSV
  → schema validation (29 features, no target leakage)
  → inference DataFrame construction
  → 24h / 48h / 72h model inference
  → prediction validation (no NaN, no zero-on-failure, AQI range)
  → Flask API /api/aqi/forecast response format
  → prediction logging to SQLite

This test catches the class of bug where predictions become 0 or 145 on failure.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.flask_api import app, calculate_us_aqi, _run_model_inference
from utils.feature_store import feature_store
from utils.db import log_prediction, get_prediction_history

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
EVAL_FILE = PROJECT_ROOT / "reports" / "model_evaluation" / "3cities" / "training_report_3cities.json"
EXPECTED_FEATURE_COUNT = 29


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    data = feature_store.load_features()
    assert not data.empty, "Feature store is empty — run the feature pipeline first"
    return data


@pytest.fixture(scope="module")
def city_df(df):
    cdf = df[df["city"].str.lower() == "lahore"].copy()
    cdf["hour"] = pd.to_datetime(cdf["hour"], errors="coerce", utc=True)
    cdf = cdf.dropna(subset=["hour"]).sort_values("hour").reset_index(drop=True)
    assert not cdf.empty, "No Lahore rows in feature store"
    return cdf


@pytest.fixture(scope="module")
def X_latest(city_df):
    excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
    feat_cols = [
        c for c in city_df.columns
        if c not in excluded and np.issubdtype(city_df[c].dtype, np.number)
    ]
    return pd.DataFrame([city_df.iloc[-1][feat_cols]])


@pytest.fixture(scope="module")
def flask_client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Stage 1: Feature store schema ─────────────────────────────────────────────

class TestFeatureStoreSchema:

    def test_feature_store_loads(self, df):
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_city_column(self, df):
        assert "city" in df.columns

    def test_all_three_cities_present(self, df):
        cities = {c.lower() for c in df["city"].unique()}
        for city in ("lahore", "islamabad", "faisalabad"):
            assert city in cities, f"City '{city}' missing from feature store"

    def test_exactly_29_numeric_features(self, city_df):
        excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
        feat_cols = [
            c for c in city_df.columns
            if c not in excluded and np.issubdtype(city_df[c].dtype, np.number)
        ]
        assert len(feat_cols) == EXPECTED_FEATURE_COUNT, (
            f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(feat_cols)}: {feat_cols}"
        )

    def test_no_target_columns_in_features(self, X_latest):
        for t in ("target_24h", "target_48h", "target_72h"):
            assert t not in X_latest.columns, f"Target '{t}' leaked into inference features"

    def test_hour_column_is_parseable_datetime(self, df):
        parsed = pd.to_datetime(df["hour"], errors="coerce", utc=True)
        null_count = parsed.isna().sum()
        total = len(parsed)
        assert null_count / total < 0.05, (
            f"More than 5% of 'hour' values could not be parsed as datetime: "
            f"{null_count}/{total}"
        )


# ── Stage 2: Preprocessing ────────────────────────────────────────────────────

class TestPreprocessing:

    def test_X_latest_is_single_row(self, X_latest):
        assert X_latest.shape[0] == 1

    def test_X_latest_has_correct_feature_count(self, X_latest):
        assert X_latest.shape[1] == EXPECTED_FEATURE_COUNT

    def test_no_inf_values_in_X(self, X_latest):
        inf_mask = np.isinf(X_latest.select_dtypes(include=[np.number]))
        assert not inf_mask.any().any(), "Infinity values found in inference input"

    def test_nan_fraction_acceptable(self, X_latest):
        """NaN is allowed — the model pipeline handles it via SimpleImputer.
        But verify it is not 100% NaN (that would indicate a data pipeline failure)."""
        nan_fraction = X_latest.isna().mean().mean()
        assert nan_fraction < 1.0, "X_latest is entirely NaN — feature pipeline failure"


# ── Stage 3: Multi-horizon inference ─────────────────────────────────────────

class TestMultiHorizonInference:

    @pytest.mark.parametrize("horizon", [24, 48, 72])
    def test_model_path_exists(self, horizon):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        assert path.exists(), f"Artifact missing: {path}"

    @pytest.mark.parametrize("horizon", [24, 48, 72])
    def test_inference_succeeds(self, horizon, X_latest):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        ok, pm25, err = _run_model_inference(path, X_latest)
        assert ok, f"+{horizon}h inference failed: {err}"
        assert pm25 >= 0.0, f"+{horizon}h predicted PM2.5 is negative: {pm25}"

    @pytest.mark.parametrize("horizon", [24, 48, 72])
    def test_predicted_pm25_is_finite(self, horizon, X_latest):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        ok, pm25, _ = _run_model_inference(path, X_latest)
        if ok:
            assert not np.isnan(pm25), f"+{horizon}h PM2.5 is NaN"
            assert not np.isinf(pm25), f"+{horizon}h PM2.5 is Inf"

    @pytest.mark.parametrize("horizon", [24, 48, 72])
    def test_derived_aqi_in_valid_range(self, horizon, X_latest):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        ok, pm25, _ = _run_model_inference(path, X_latest)
        if ok:
            aqi = calculate_us_aqi(pm25)
            assert 0 <= aqi <= 500, f"+{horizon}h AQI {aqi} out of valid range"

    def test_zero_prediction_detection(self, X_latest):
        """
        Regression test: catches the bug where all three forecasts return 0.
        If a model inference failure is masked as AQI=0, this test catches it.
        """
        predictions = {}
        for horizon in [24, 48, 72]:
            path = MODEL_DIR / f"best_model_{horizon}h.joblib"
            ok, pm25, err = _run_model_inference(path, X_latest)
            if ok:
                predictions[horizon] = calculate_us_aqi(pm25)

        if len(predictions) == 3:
            # All three predictions succeeded — none should be exactly zero
            # unless PM2.5 is genuinely 0, which is extremely unlikely in Pakistan
            zero_count = sum(1 for v in predictions.values() if v == 0)
            assert zero_count < 3, (
                f"All three forecasts returned AQI=0. This indicates a model inference "
                f"failure being silently converted to zero. Predictions: {predictions}"
            )

    def test_forecasts_are_not_all_identical(self, X_latest):
        """Three distinct horizon models must produce at least two different outputs."""
        predictions = {}
        for horizon in [24, 48, 72]:
            path = MODEL_DIR / f"best_model_{horizon}h.joblib"
            ok, pm25, _ = _run_model_inference(path, X_latest)
            if ok:
                predictions[horizon] = round(pm25, 3)

        if len(predictions) == 3:
            unique_preds = set(predictions.values())
            assert len(unique_preds) > 1, (
                f"All three horizon models returned identical PM2.5={list(unique_preds)[0]}. "
                "This suggests the same model artifact is being loaded for all horizons."
            )


# ── Stage 4: Flask API response contract ──────────────────────────────────────

class TestFlaskForecastContract:

    def test_forecast_endpoint_returns_200(self, flask_client):
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        assert r.status_code == 200

    def test_forecast_response_has_success_flag(self, flask_client):
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        assert data.get("success") is True

    def test_forecast_response_has_all_horizons(self, flask_client):
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        forecast = data.get("forecast", {})
        assert "24h" in forecast
        assert "48h" in forecast
        assert "72h" in forecast

    def test_forecast_horizon_has_status_key(self, flask_client):
        """Each horizon in the forecast must have a 'status' key — never silently missing."""
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        for h in ("24h", "48h", "72h"):
            entry = data["forecast"][h]
            assert "status" in entry, f"Forecast horizon {h} missing 'status' key"

    def test_successful_forecast_has_aqi_field(self, flask_client):
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        for h in ("24h", "48h", "72h"):
            entry = data["forecast"][h]
            if entry.get("status") == "success":
                assert "aqi" in entry, f"Successful forecast for {h} missing 'aqi'"
                assert isinstance(entry["aqi"], int)
                assert 0 <= entry["aqi"] <= 500

    def test_successful_forecast_has_category(self, flask_client):
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        for h in ("24h", "48h", "72h"):
            entry = data["forecast"][h]
            if entry.get("status") == "success":
                assert "category" in entry
                assert entry["category"] not in ("", None)

    def test_forecast_aqi_never_zero_on_failure(self, flask_client):
        """Regression: a failed forecast must have status='unavailable', not aqi=0."""
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        for h in ("24h", "48h", "72h"):
            entry = data["forecast"][h]
            if entry.get("status") == "unavailable":
                assert "aqi" not in entry, (
                    f"Unavailable forecast for {h} should NOT have an 'aqi' field. "
                    "A missing/failed forecast must not be displayed as AQI=0."
                )

    def test_unsupported_city_returns_400(self, flask_client):
        r = flask_client.get("/api/aqi/forecast?city=Karachi")
        assert r.status_code == 400
        data = r.get_json()
        assert data["success"] is False

    def test_data_source_field_present(self, flask_client):
        r = flask_client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        assert "data_source" in data


# ── Stage 5: Prediction logging ───────────────────────────────────────────────

class TestPredictionLogging:

    def test_log_prediction_persists_to_db(self):
        ok = log_prediction("Lahore", 185, 172, 160, user_id=None, model_version="v2.1.0-test")
        assert ok is True

    def test_logged_prediction_appears_in_history(self):
        history = get_prediction_history(user_id=None, limit=10)
        assert len(history) > 0

    def test_history_has_required_fields(self):
        history = get_prediction_history(user_id=None, limit=5)
        required = {"city", "predicted_aqi_24h", "predicted_aqi_48h", "predicted_aqi_72h", "timestamp"}
        for row in history:
            for field in required:
                assert field in row, f"History row missing field: {field}"

    def test_logged_aqi_values_are_integers(self):
        ok = log_prediction("Islamabad", 210, 198, 184)
        assert ok is True
        history = get_prediction_history(limit=5)
        for row in history:
            for key in ("predicted_aqi_24h", "predicted_aqi_48h", "predicted_aqi_72h"):
                val = row.get(key)
                if val is not None:
                    assert isinstance(val, int), f"{key} should be int, got {type(val)}: {val}"
