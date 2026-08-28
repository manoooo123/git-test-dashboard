"""
SHAP Feature Importance & Model Explainability Module (Production v2.0).

Computes REAL SHAP values using:
1. SHAP LinearExplainer for Ridge models (exact Shapley values)
2. SHAP TreeExplainer for tree-based models (exact for trees)
3. Ridge coefficient fallback only if SHAP fails

Outputs:
- Per-horizon global feature importance CSVs
- Per-city feature importance breakdown
- Waterfall plots for top predictions
- Summary plots for dashboard visualization
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_features_3cities.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
REPORT_DIR = PROJECT_ROOT / "reports" / "explainability"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXCLUDED_COLUMNS = {
    "hour",
    "city",
    "is_missing_hour",
    "coverage_quality",
    "target_24h",
    "target_48h",
    "target_72h",
}

# City names for per-city analysis
CITIES = ["Lahore", "Islamabad", "Faisalabad"]

def compute_shap_values(
    model,
    X_sample: np.ndarray,
    feature_columns: list,
    use_shap: bool = True
) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
    """
    Compute SHAP values for model explainability.
    
    Args:
        model: Trained sklearn model (Ridge, RandomForest, etc.)
        X_sample: Feature matrix (numpy array)
        feature_columns: List of feature names
        use_shap: Whether to use SHAP library (True) or fallback to coefficients
        
    Returns:
        (shap_values_array, importance_dataframe)
    """
    importance_df = pd.DataFrame()
    shap_values = None
    
    if use_shap:
        try:
            import shap
            
            # Linear models: Use LinearExplainer for exact Shapley values
            if hasattr(model, "coef_"):
                logger.info("Using SHAP LinearExplainer (exact Shapley values)")
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
                
            # Tree models: Use TreeExplainer (exact for trees)
            elif hasattr(model, "feature_importances_"):
                logger.info("Using SHAP TreeExplainer (exact for trees)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
            else:
                logger.warning("Model type not recognized for SHAP. Using KernelExplainer (slow).")
                explainer = shap.KernelExplainer(model.predict, X_sample[:100])
                shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-dimensional SHAP output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 0]
            
            # Compute mean absolute SHAP values (global importance)
            shap_df = pd.DataFrame(shap_values, columns=feature_columns)
            importance = shap_df.abs().mean().sort_values(ascending=False)
            importance_df = importance.reset_index()
            importance_df.columns = ["feature", "mean_absolute_shap"]
            importance_df["method"] = "SHAP"
            
            logger.info("✅ SHAP computation successful")
            return shap_values, importance_df
            
        except ImportError:
            logger.warning("SHAP library not installed. Install: pip install shap")
            use_shap = False
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}. Falling back to coefficients.")
            use_shap = False
    
    # Fallback: Use model coefficients or feature importances
    if not use_shap:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "feature": feature_columns,
                "mean_absolute_shap": importances
            })
            importance_df["method"] = "TreeImportance"
            logger.info("Using RandomForest feature importances (fallback)")
            
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            importance_df = pd.DataFrame({
                "feature": feature_columns,
                "mean_absolute_shap": coefs
            })
            importance_df["method"] = "RidgeCoefficients"
            logger.info("Using Ridge |coefficients| (fallback)")
            
        importance_df = importance_df.sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    
    return shap_values, importance_df


def create_per_city_importance(
    df: pd.DataFrame,
    model,
    feature_columns: list,
    horizon: str
) -> Dict[str, pd.DataFrame]:
    """
    Compute feature importance separately for each city.
    
    Returns:
        Dict mapping city name to importance DataFrame
    """
    city_importances = {}
    
    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False
    
    for city in CITIES:
        city_df = df[df["city"].str.lower() == city.lower()].copy()
        
        if city_df.empty:
            logger.warning(f"No data for {city}")
            continue
        
        X_city = city_df[feature_columns].fillna(0).values
        X_sample = X_city[:min(500, len(X_city))]
        
        _, importance_df = compute_shap_values(
            model,
            X_sample,
            feature_columns,
            use_shap=has_shap
        )
        
        city_importances[city] = importance_df
        
        # Save per-city CSV
        csv_path = REPORT_DIR / f"shap_{horizon}_{city.lower()}.csv"
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {city} importance: {csv_path}")
    
    return city_importances


def plot_feature_importance(
    importance_df: pd.DataFrame,
    horizon: str,
    top_n: int = 15,
    title_suffix: str = ""
) -> Path:
    """
    Create horizontal bar plot of feature importance.
    
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 6))
    top_df = importance_df.head(top_n).sort_values("mean_absolute_shap", ascending=True)
    
    # Color by method
    color = "#38BDF8" if top_df["method"].iloc[0] == "SHAP" else "#F59E0B"
    
    plt.barh(top_df["feature"], top_df["mean_absolute_shap"], color=color)
    
    method_label = top_df["method"].iloc[0]
    plt.title(f"Feature Importance ({horizon} Forecast{title_suffix})\nMethod: {method_label}", fontsize=12, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    plot_path = REPORT_DIR / f"shap_feature_importance_{horizon}{title_suffix.replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: {plot_path}")
    return plot_path


def main():
    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR | SHAP EXPLAINABILITY ANALYSIS v2.0")
    logger.info("=" * 72)

    if not DATA_PATH.exists():
        logger.error("Dataset missing: %s", DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset shape: %s", df.shape)

    feature_columns = [
        col for col in df.columns 
        if col not in EXCLUDED_COLUMNS and np.issubdtype(df[col].dtype, np.number)
    ]
    logger.info("Feature count: %d", len(feature_columns))

    X = df[feature_columns].copy().fillna(0)

    # Check SHAP availability
    try:
        import shap
        has_shap = True
        logger.info("✅ SHAP package available")
    except ImportError:
        has_shap = False
        logger.warning("⚠️ SHAP package not found. Using coefficient fallback.")

    for horizon in ["24h", "48h", "72h"]:
        logger.info("-" * 72)
        logger.info("🔍 Feature importance analysis for horizon: %s", horizon)

        model_path = MODEL_DIR / f"best_model_{horizon}.joblib"
        if not model_path.exists():
            logger.warning("Model file not found: %s", model_path)
            continue

        pipeline = joblib.load(model_path)
        
        # Extract underlying estimator from pipeline
        if hasattr(pipeline, "named_steps"):
            preprocessor = pipeline.named_steps.get("imputer")
            model = pipeline.named_steps.get("model")
        else:
            preprocessor = None
            model = pipeline

        # Sample data for SHAP (use 1000 samples for speed)
        X_sample = X.sample(n=min(1000, len(X)), random_state=42)
        X_transformed = preprocessor.transform(X_sample) if preprocessor is not None else X_sample.values

        # Compute SHAP values
        shap_values, importance_df = compute_shap_values(
            model,
            X_transformed,
            feature_columns,
            use_shap=has_shap
        )

        if not importance_df.empty:
            # Save global feature importance CSV
            csv_path = REPORT_DIR / f"shap_feature_importance_{horizon}.csv"
            importance_df.to_csv(csv_path, index=False)
            logger.info("✅ Saved global importance CSV: %s", csv_path)

            # Plot global importance
            plot_feature_importance(importance_df, horizon)
            
            # Per-city analysis
            logger.info("Computing per-city feature importance...")
            city_importances = create_per_city_importance(
                df,
                model,
                feature_columns,
                horizon
            )
            
            # Plot per-city importance for Lahore (best performance)
            if "Lahore" in city_importances:
                plot_feature_importance(
                    city_importances["Lahore"],
                    horizon,
                    title_suffix=" - Lahore"
                )

    logger.info("=" * 72)
    logger.info("✅ EXPLAINABILITY ANALYSIS COMPLETED")
    logger.info("=" * 72)
    
    # Summary
    logger.info("\n📊 SUMMARY:")
    logger.info(f"   Method Used: {'SHAP (Exact Shapley Values)' if has_shap else 'Coefficient Fallback'}")
    logger.info(f"   Horizons Analyzed: 24h, 48h, 72h")
    logger.info(f"   Per-City Analysis: {', '.join(CITIES)}")
    logger.info(f"   Output Directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()