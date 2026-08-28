"""
Authoritative Feature Contract for Pearls AQI Predictor

This module defines the exact 29 features that trained models expect.
All production inference must use exactly these features.
"""

# Authoritative model feature contract (29 features)
# DO NOT modify without retraining all models
MODEL_FEATURES = [
    "sensor_count",
    "pm25_mean", 
    "pm25_median",
    "pm25_max",
    "temperature",
    "humidity", 
    "pressure",
    "wind_speed",
    "clouds",
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h", 
    "pm25_lag_12h",
    "pm25_lag_24h",
    "hour_of_day",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "pm25_rolling_mean_3h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_std_24h", 
    "pm25_change_1h",
    "pm25_change_24h"
]

# Features to exclude from model input (metadata/targets)
EXCLUDED_FEATURES = {
    "city", "hour", "timestamp",
    "coverage_quality", "is_missing_hour",  # Post-training metadata
    "target_24h", "target_48h", "target_72h",
    "target_pm2_5_24h", "target_pm2_5_48h", "target_pm2_5_72h"
}

def get_model_features(df):
    """Extract exactly the 29 features that models expect."""
    available_features = set(df.columns) - EXCLUDED_FEATURES
    missing_features = set(MODEL_FEATURES) - available_features
    
    if missing_features:
        raise ValueError(f"Missing required model features: {missing_features}")
    
    return df[MODEL_FEATURES]

def validate_feature_schema(df):
    """Validate that dataframe contains all required model features."""
    available_features = set(df.columns) - EXCLUDED_FEATURES
    missing_features = set(MODEL_FEATURES) - available_features
    extra_features = available_features - set(MODEL_FEATURES)
    
    return {
        "valid": len(missing_features) == 0,
        "missing": list(missing_features),
        "extra": list(extra_features),
        "expected_count": len(MODEL_FEATURES),
        "actual_count": len(available_features)
    }