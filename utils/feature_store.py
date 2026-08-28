"""
Pearls AQI Predictor - Feature Store Management Module (Hopsworks & Local Offline Store).

Provides a unified interface to:
1. Connect to Hopsworks cloud feature store when HOPSWORKS_API_KEY is configured.
2. Read and write AQI feature groups across Lahore, Islamabad, and Faisalabad.
3. Provide robust offline local feature store fallback (data/processed/model_features_3cities.csv).
4. Supply feature store telemetry and health metrics to the application dashboard and REST API.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "model_features_3cities.csv"
FEATURE_GROUP_NAME = "aqi_hourly_features_3cities"
FEATURE_GROUP_VERSION = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pearls_feature_store")


class FeatureStoreManager:
    """Enterprise Feature Store Manager for Hopsworks & Local Storage."""

    def __init__(self, project_name: str = "pearls_aqi_predictor"):
        self.project_name = os.getenv("HOPSWORKS_PROJECT_NAME", project_name)
        self.api_key = os.getenv("HOPSWORKS_API_KEY")
        self.connected_to_hopsworks = False
        self.project = None
        self.fs = None

        if self.api_key:
            self._connect_hopsworks()

    def _connect_hopsworks(self) -> bool:
        """Attempt connection to Hopsworks Cloud Feature Store."""
        try:
            import hopsworks
            logger.info("Attempting connection to Hopsworks Feature Store...")
            project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = project.get_feature_store()
            self.project = project
            self.connected_to_hopsworks = True
            logger.info("Successfully connected to Hopsworks Feature Store!")
            return True
        except ImportError:
            logger.warning("Hopsworks package not installed. Using local feature store.")
        except Exception as e:
            logger.warning(f"Could not connect to Hopsworks ({e}). Using local offline feature store.")
        self.connected_to_hopsworks = False
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return feature store telemetry and health status."""
        local_exists = LOCAL_FEATURE_FILE.exists()
        record_count = 0
        feature_count = 0
        last_modified = None

        if local_exists:
            try:
                df = pd.read_csv(LOCAL_FEATURE_FILE, nrows=5)
                feature_count = len(df.columns)
                # Count total rows safely
                with open(LOCAL_FEATURE_FILE, "r", encoding="utf-8") as f:
                    record_count = max(0, sum(1 for _ in f) - 1)
                last_modified = pd.Timestamp(LOCAL_FEATURE_FILE.stat().st_mtime, unit="s").isoformat()
            except Exception:
                pass

        return {
            "hopsworks_configured": bool(self.api_key),
            "hopsworks_connected": self.connected_to_hopsworks,
            "active_store_type": "Hopsworks Cloud Store" if self.connected_to_hopsworks else "Local Offline Feature Store",
            "local_store_available": local_exists,
            "feature_group_name": FEATURE_GROUP_NAME,
            "version": FEATURE_GROUP_VERSION,
            "record_count": record_count,
            "feature_count": feature_count,
            "last_updated": last_modified or "N/A"
        }

    def load_features(self) -> pd.DataFrame:
        """Load feature store dataset for model inference and training."""
        if self.connected_to_hopsworks and self.fs is not None:
            try:
                fg = self.fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
                df = fg.read()
                logger.info(f"Loaded {len(df)} records from Hopsworks Feature Group '{FEATURE_GROUP_NAME}'.")
                return df
            except Exception as e:
                logger.error(f"Failed reading from Hopsworks Feature Group: {e}. Falling back to local file.")

        # Local offline feature store fallback
        if LOCAL_FEATURE_FILE.exists():
            logger.info(f"Loading features from local feature store: {LOCAL_FEATURE_FILE}")
            df = pd.read_csv(LOCAL_FEATURE_FILE)
            if "hour" in df.columns:
                df["hour"] = pd.to_datetime(df["hour"], errors="coerce", utc=True)
                df = df.sort_values(["city", "hour"]).reset_index(drop=True)
            return df
        else:
            logger.error(f"Local feature store file not found: {LOCAL_FEATURE_FILE}")
            return pd.DataFrame()

    def save_features(self, df: pd.DataFrame) -> bool:
        """Save updated feature matrix to Hopsworks and local storage."""
        if df.empty:
            logger.warning("Attempted to save empty DataFrame to Feature Store.")
            return False

        saved_local = False
        try:
            LOCAL_FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(LOCAL_FEATURE_FILE, index=False)
            logger.info(f"Saved {len(df)} feature records to local feature store.")
            saved_local = True
        except Exception as e:
            logger.error(f"Error saving local feature store: {e}")

        if self.connected_to_hopsworks and self.fs is not None:
            try:
                fg = self.fs.get_or_create_feature_group(
                    name=FEATURE_GROUP_NAME,
                    version=FEATURE_GROUP_VERSION,
                    primary_key=["city", "hour"],
                    description="Hourly AQI and atmospheric features for Lahore, Islamabad, and Faisalabad"
                )
                fg.insert(df)
                logger.info(f"Successfully inserted {len(df)} records into Hopsworks Feature Group.")
            except Exception as e:
                logger.error(f"Error inserting into Hopsworks Feature Group: {e}")

        return saved_local


# Global singleton instance
feature_store = FeatureStoreManager()
