"""
Live SHAP Explainability for Real-Time Predictions.

Provides real-time feature importance explanations for individual predictions
in the Streamlit dashboard and Flask API.

Key Features:
- Instant SHAP value computation for single predictions
- Top-K most influential features identification
- Positive/negative contribution breakdown
- Support for both global and local explanations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
REPORT_DIR = PROJECT_ROOT / "reports" / "explainability"

logger = logging.getLogger(__name__)


class LiveExplainer:
    """
    Real-time SHAP explainability for AQI predictions.
    
    Caches explainers per horizon to avoid recomputation overhead.
    """
    
    def __init__(self):
        self.explainers: Dict[str, any] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.baseline_values: Dict[str, np.ndarray] = {}
        self.has_shap = self._check_shap()
        
    def _check_shap(self) -> bool:
        """Check if SHAP library is available."""
        try:
            import shap
            return True
        except ImportError:
            logger.warning("SHAP library not installed. Explainability will use coefficient fallback.")
            return False
    
    def _load_explainer(self, horizon: str) -> bool:
        """
        Load and cache SHAP explainer for a horizon.
        
        Args:
            horizon: "24h", "48h", or "72h"
            
        Returns:
            True if explainer loaded successfully
        """
        if horizon in self.explainers:
            return True
        
        model_path = MODEL_DIR / f"best_model_{horizon}.joblib"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        try:
            pipeline = joblib.load(model_path)
            
            # Extract model from pipeline
            if hasattr(pipeline, "named_steps"):
                model = pipeline.named_steps.get("model")
            else:
                model = pipeline
            
            if model is None:
                logger.error(f"Could not extract model from pipeline for {horizon}")
                return False
            
            # Load global feature importance to get feature columns
            importance_csv = REPORT_DIR / f"shap_feature_importance_{horizon}.csv"
            if importance_csv.exists():
                importance_df = pd.read_csv(importance_csv)
                self.feature_columns[horizon] = importance_df["feature"].tolist()
            else:
                logger.warning(f"Feature importance CSV not found: {importance_csv}")
                return False
            
            # Create explainer
            if self.has_shap:
                import shap
                
                if hasattr(model, "coef_"):
                    # Linear model - use LinearExplainer
                    # Create a dummy background (zero vector) for fast computation
                    background = np.zeros((1, len(self.feature_columns[horizon])))
                    explainer = shap.LinearExplainer(model, background)
                    self.explainers[horizon] = explainer
                    self.baseline_values[horizon] = background[0]
                    logger.info(f"Loaded LinearExplainer for {horizon}")
                    
                elif hasattr(model, "feature_importances_"):
                    # Tree model - use TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    self.explainers[horizon] = explainer
                    self.baseline_values[horizon] = explainer.expected_value
                    logger.info(f"Loaded TreeExplainer for {horizon}")
                    
                else:
                    logger.warning(f"Unsupported model type for {horizon}")
                    # Store model for coefficient fallback
                    self.explainers[horizon] = model
                    
            else:
                # No SHAP - store model for coefficient fallback
                self.explainers[horizon] = model
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading explainer for {horizon}: {e}")
            return False
    
    def explain_prediction(
        self,
        X: np.ndarray,
        horizon: str,
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X: Feature vector (1D array)
            horizon: "24h", "48h", or "72h"
            top_k: Number of top features to return
            
        Returns:
            Dict with:
            - top_features: List of (feature_name, shap_value, feature_value)
            - positive_contributors: Features increasing prediction
            - negative_contributors: Features decreasing prediction
            - baseline: Expected value (no features)
            - prediction: Model prediction
            - method: "SHAP" or "Coefficients"
        """
        # Load explainer if not cached
        if not self._load_explainer(horizon):
            return {
                "error": f"Could not load explainer for {horizon}",
                "method": "unavailable"
            }
        
        explainer = self.explainers[horizon]
        feature_names = self.feature_columns[horizon]
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Verify feature count matches
        if X.shape[1] != len(feature_names):
            logger.error(f"Feature count mismatch: X has {X.shape[1]}, expected {len(feature_names)}")
            return {"error": "Feature count mismatch", "method": "unavailable"}
        
        try:
            if self.has_shap:
                import shap
                
                # Compute SHAP values
                if isinstance(explainer, shap.Explainer):
                    shap_values = explainer.shap_values(X)
                    
                    # Handle multi-dimensional output
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    if hasattr(shap_values, "values"):
                        shap_values = shap_values.values
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                    
                    baseline = self.baseline_values.get(horizon, 0)
                    prediction = float(baseline + shap_values.sum())
                    method = "SHAP"
                    
                else:
                    # Fallback: Use model coefficients
                    shap_values = self._compute_coefficient_importance(explainer, X[0], feature_names)
                    baseline = 0
                    prediction = float(explainer.predict(X)[0])
                    method = "Coefficients"
            else:
                # Fallback: Use model coefficients
                shap_values = self._compute_coefficient_importance(explainer, X[0], feature_names)
                baseline = 0
                prediction = float(explainer.predict(X)[0])
                method = "Coefficients"
            
            # Create feature-value-importance tuples
            feature_impacts = []
            for fname, fval, shap_val in zip(feature_names, X[0], shap_values):
                feature_impacts.append({
                    "feature": fname,
                    "value": float(fval),
                    "impact": float(shap_val),
                    "abs_impact": abs(float(shap_val))
                })
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
            
            # Split into positive and negative contributors
            positive = [f for f in feature_impacts if f["impact"] > 0][:top_k]
            negative = [f for f in feature_impacts if f["impact"] < 0][:top_k]
            
            return {
                "method": method,
                "baseline": float(baseline) if isinstance(baseline, (int, float, np.number)) else float(baseline[0]) if hasattr(baseline, '__getitem__') else 0.0,
                "prediction": prediction,
                "top_features": feature_impacts[:top_k],
                "positive_contributors": positive,
                "negative_contributors": negative,
            }
            
        except Exception as e:
            logger.error(f"Error computing SHAP values for {horizon}: {e}")
            return {"error": str(e), "method": "unavailable"}
    
    def _compute_coefficient_importance(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Fallback: Compute feature importance using model coefficients.
        
        For linear models: importance = coefficient * feature_value
        For tree models: importance = feature_importance (global, not instance-specific)
        """
        if hasattr(model, "coef_"):
            # Linear model: coefficient * feature value
            coefs = model.coef_
            return coefs * X
        elif hasattr(model, "feature_importances_"):
            # Tree model: use global importances (not instance-specific)
            return model.feature_importances_ * X
        else:
            # Unknown model type
            return np.zeros_like(X)
    
    def get_global_importance(self, horizon: str, top_k: int = 15) -> List[Dict[str, any]]:
        """
        Get global feature importance from pre-computed SHAP CSV.
        
        Args:
            horizon: "24h", "48h", or "72h"
            top_k: Number of top features to return
            
        Returns:
            List of {feature, importance, rank}
        """
        importance_csv = REPORT_DIR / f"shap_feature_importance_{horizon}.csv"
        
        if not importance_csv.exists():
            logger.warning(f"Global importance CSV not found: {importance_csv}")
            return []
        
        try:
            df = pd.read_csv(importance_csv).head(top_k)
            return [
                {
                    "feature": row["feature"],
                    "importance": float(row["mean_absolute_shap"]),
                    "rank": idx + 1
                }
                for idx, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error reading global importance: {e}")
            return []


# Global singleton
live_explainer = LiveExplainer()
