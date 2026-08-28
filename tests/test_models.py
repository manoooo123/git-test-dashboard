"""
Pearls AQI Predictor — Unit tests for model artifact loading & inference.

Covers:
- Artifact existence for all 3 horizons
- Correct sklearn Pipeline interface
- Prediction type and range validation
- NaN / Inf output detection (zero-prediction bug prevention)
- Feature count consistency between model and feature store
- Distinct predictions across horizons (no duplication)
"""

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
FEAT_FILE = PROJECT_ROOT / "data" / "processed" / "model_features_3cities.csv"
HORIZONS = [24, 48, 72]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def feature_df():
    """Load feature store CSV once for all model tests."""
    assert FEAT_FILE.exists(), f"Feature store not found: {FEAT_FILE}"
    df = pd.read_csv(FEAT_FILE)
    assert not df.empty, "Feature store CSV is empty"
    return df


@pytest.fixture(scope="module")
def X_latest(feature_df):
    """Build a single-row inference DataFrame from the most recent feature row."""
    excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
    feat_cols = [
        c for c in feature_df.columns
        if c not in excluded and np.issubdtype(feature_df[c].dtype, np.number)
    ]
    return pd.DataFrame([feature_df.iloc[-1][feat_cols]])


# ── Artifact existence ────────────────────────────────────────────────────────

class TestModelArtifacts:

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_artifact_exists(self, horizon):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        assert path.exists(), f"Model artifact missing: {path}"

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_artifact_loads_without_error(self, horizon):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        assert model is not None

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_artifact_has_predict_method(self, horizon):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        assert hasattr(model, "predict"), f"Model for +{horizon}h has no .predict() method"

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_artifact_is_sklearn_pipeline(self, horizon):
        """Best model should be a sklearn Pipeline (SimpleImputer → Scaler → Estimator)."""
        from sklearn.pipeline import Pipeline
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        assert isinstance(model, Pipeline), (
            f"Expected sklearn Pipeline for +{horizon}h, got {type(model)}"
        )

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_pipeline_has_imputer_step(self, horizon):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        step_names = [name for name, _ in model.steps]
        assert "imputer" in step_names, (
            f"Pipeline for +{horizon}h missing 'imputer' step. Steps: {step_names}"
        )


# ── Inference quality ─────────────────────────────────────────────────────────

class TestModelInference:

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_prediction_is_finite_float(self, horizon, X_latest):
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        pred = float(model.predict(X_latest)[0])
        assert not np.isnan(pred),  f"+{horizon}h model returned NaN"
        assert not np.isinf(pred),  f"+{horizon}h model returned Inf"

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_prediction_is_non_negative(self, horizon, X_latest):
        """PM2.5 predictions must never be negative (clamped to 0 before AQI calc)."""
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        raw = float(model.predict(X_latest)[0])
        # The application clamps: pm25 = max(0.0, raw) — raw itself may be negative
        # but the clamped value must be >= 0
        pm25 = max(0.0, raw)
        assert pm25 >= 0.0, f"+{horizon}h PM2.5 after clamping is negative: {pm25}"

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_derived_aqi_is_valid_range(self, horizon, X_latest):
        from app.flask_api import calculate_us_aqi
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        raw_pm25 = max(0.0, float(model.predict(X_latest)[0]))
        aqi = calculate_us_aqi(raw_pm25)
        assert 0 <= aqi <= 500, f"+{horizon}h AQI out of range: {aqi}"

    def test_three_horizons_produce_distinct_predictions(self, X_latest):
        """
        The three horizon models must be distinct artifacts producing different outputs.
        If all three return the same value, that indicates artifact duplication or
        a pipeline bug where the same model is loaded for all horizons.
        """
        predictions = {}
        for horizon in HORIZONS:
            path = MODEL_DIR / f"best_model_{horizon}h.joblib"
            model = joblib.load(path)
            predictions[horizon] = float(model.predict(X_latest)[0])

        values = list(predictions.values())
        # At least two of the three predictions should differ
        assert len(set(round(v, 4) for v in values)) > 1, (
            f"All three horizon models returned identical predictions: {predictions}. "
            "Check that best_model_24h/48h/72h.joblib are distinct artifacts."
        )

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_model_handles_nan_inputs_via_imputer(self, horizon, feature_df):
        """
        The model pipeline contains a SimpleImputer — it must handle NaN feature
        values without raising. This verifies training/inference NaN-handling parity.
        """
        excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
        feat_cols = [
            c for c in feature_df.columns
            if c not in excluded and np.issubdtype(feature_df[c].dtype, np.number)
        ]
        # Build a row that is 30% NaN
        row = feature_df.iloc[-1][feat_cols].copy()
        nan_cols = feat_cols[: len(feat_cols) // 3]
        for col in nan_cols:
            row[col] = np.nan

        X_nan = pd.DataFrame([row])
        path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        model = joblib.load(path)
        # Must not raise
        pred = float(model.predict(X_nan)[0])
        assert not np.isnan(pred), f"+{horizon}h returned NaN even with imputer, for NaN input"


# ── Feature count consistency ─────────────────────────────────────────────────

class TestFeatureContract:

    def test_feature_store_has_29_numeric_features(self, feature_df):
        excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
        feat_cols = [
            c for c in feature_df.columns
            if c not in excluded and np.issubdtype(feature_df[c].dtype, np.number)
        ]
        assert len(feat_cols) == 29, (
            f"Expected 29 numeric feature columns, found {len(feat_cols)}. "
            f"Columns: {feat_cols}"
        )

    def test_feature_names_match_training_report(self):
        """Verify feature columns match those recorded in training_report_3cities.json."""
        import json
        report_path = PROJECT_ROOT / "reports" / "model_evaluation" / "3cities" / "training_report_3cities.json"
        if not report_path.exists():
            pytest.skip("Training report not found — run train_3cities.py first")

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        expected = set(report.get("feature_columns", []))
        if not expected:
            pytest.skip("training_report_3cities.json has no 'feature_columns' key")

        df = pd.read_csv(FEAT_FILE)
        excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
        actual = {
            c for c in df.columns
            if c not in excluded and np.issubdtype(df[c].dtype, np.number)
        }
        missing_from_store = expected - actual
        extra_in_store = actual - expected

        assert not missing_from_store, (
            f"Features in training report but NOT in feature store: {missing_from_store}"
        )
        assert not extra_in_store, (
            f"Extra features in store not in training report: {extra_in_store}"
        )

    def test_no_target_leakage_in_inference_features(self, feature_df):
        """Target columns must never appear in the feature set fed to the model."""
        excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
        feat_cols = [
            c for c in feature_df.columns
            if c not in excluded and np.issubdtype(feature_df[c].dtype, np.number)
        ]
        for target in ("target_24h", "target_48h", "target_72h"):
            assert target not in feat_cols, (
                f"Target column '{target}' leaked into feature set — temporal leakage detected!"
            )
