"""
Model Health & Performance Monitoring Module.

Provides real-time model artifact status and training metrics for:
- Model availability (all 3 horizons)
- Model performance (MAE, RMSE, R²)
- Model version tracking
- Training data statistics
- Model freshness

Used by Dashboard and Flask API to expose actual model health.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
TRAINING_REPORT = PROJECT_ROOT / "reports" / "model_evaluation" / "3cities" / "training_report_3cities.json"
SHAP_DIR = PROJECT_ROOT / "reports" / "explainability"

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Model artifact status."""
    
    horizon: str
    available: bool
    path: Optional[str]
    size_mb: Optional[float]
    last_modified: Optional[str]
    error: Optional[str]


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    
    horizon: str
    model_type: str
    mae: float
    rmse: float
    r2: float
    samples_trained: int
    by_city: Dict[str, Dict[str, float]]


@dataclass
class ModelHealthReport:
    """Complete model health assessment."""
    
    overall_status: str  # "healthy", "degraded", "offline"
    models_available: int
    models_expected: int
    model_statuses: List[ModelStatus]
    performance_metrics: List[ModelPerformance]
    training_info: Dict[str, Any]
    shap_available: bool
    last_training: Optional[str]


class ModelHealthMonitor:
    """Real-time model health monitoring."""
    
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        training_report: Path = TRAINING_REPORT,
        shap_dir: Path = SHAP_DIR
    ):
        self.model_dir = model_dir
        self.training_report = training_report
        self.shap_dir = shap_dir
        
    def get_health_report(self) -> ModelHealthReport:
        """
        Generate comprehensive model health report.
        
        Returns:
            ModelHealthReport with real artifact status and performance
        """
        horizons = ["24h", "48h", "72h"]
        
        # Check model artifacts
        model_statuses = []
        available_count = 0
        
        for horizon in horizons:
            model_path = self.model_dir / f"best_model_{horizon}.joblib"
            
            if model_path.exists():
                try:
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    last_modified = datetime.fromtimestamp(
                        model_path.stat().st_mtime
                    ).isoformat()
                    
                    # Try loading to verify integrity
                    joblib.load(model_path)
                    
                    model_statuses.append(ModelStatus(
                        horizon=horizon,
                        available=True,
                        path=str(model_path),
                        size_mb=round(size_mb, 2),
                        last_modified=last_modified,
                        error=None
                    ))
                    available_count += 1
                    
                except Exception as e:
                    model_statuses.append(ModelStatus(
                        horizon=horizon,
                        available=False,
                        path=str(model_path),
                        size_mb=None,
                        last_modified=None,
                        error=f"Load error: {str(e)}"
                    ))
            else:
                model_statuses.append(ModelStatus(
                    horizon=horizon,
                    available=False,
                    path=str(model_path),
                    size_mb=None,
                    last_modified=None,
                    error="Model file not found"
                ))
        
        # Load performance metrics
        performance_metrics = []
        training_info = {}
        last_training = None
        
        if self.training_report.exists():
            try:
                with open(self.training_report, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                training_info = {
                    "dataset_rows": report.get("dataset_rows", 0),
                    "feature_count": report.get("feature_count", 0),
                    "test_size": report.get("test_size", 0.2),
                }
                
                results = report.get("results", {})
                
                for horizon in horizons:
                    h_key = horizon
                    if h_key in results:
                        h_data = results[h_key]
                        best_model = h_data.get("best_model", "Unknown")
                        best_metrics = h_data.get("best_metrics", {})
                        best_by_city = h_data.get("best_by_city", {})
                        
                        performance_metrics.append(ModelPerformance(
                            horizon=horizon,
                            model_type=best_model,
                            mae=best_metrics.get("MAE", 0.0),
                            rmse=best_metrics.get("RMSE", 0.0),
                            r2=best_metrics.get("R2", 0.0),
                            samples_trained=h_data.get("train_samples", 0),
                            by_city=best_by_city
                        ))
                
                # Get last training timestamp from file modification
                last_training = datetime.fromtimestamp(
                    self.training_report.stat().st_mtime
                ).isoformat()
                
            except Exception as e:
                logger.error(f"Error loading training report: {e}")
        
        # Check SHAP availability
        shap_available = all(
            (self.shap_dir / f"shap_feature_importance_{h}.csv").exists()
            for h in horizons
        )
        
        # Determine overall status
        if available_count == len(horizons):
            overall_status = "healthy"
        elif available_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "offline"
        
        return ModelHealthReport(
            overall_status=overall_status,
            models_available=available_count,
            models_expected=len(horizons),
            model_statuses=model_statuses,
            performance_metrics=performance_metrics,
            training_info=training_info,
            shap_available=shap_available,
            last_training=last_training
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get quick performance summary for dashboard."""
        report = self.get_health_report()
        
        summary = {
            "status": report.overall_status,
            "models_ready": f"{report.models_available}/{report.models_expected}",
            "last_training": report.last_training,
        }
        
        # Add average metrics
        if report.performance_metrics:
            avg_mae = sum(p.mae for p in report.performance_metrics) / len(report.performance_metrics)
            avg_rmse = sum(p.rmse for p in report.performance_metrics) / len(report.performance_metrics)
            avg_r2 = sum(p.r2 for p in report.performance_metrics) / len(report.performance_metrics)
            
            summary["avg_metrics"] = {
                "MAE": round(avg_mae, 2),
                "RMSE": round(avg_rmse, 2),
                "R²": round(avg_r2, 3)
            }
        
        return summary
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get model comparison across horizons for analytics."""
        report = self.get_health_report()
        
        comparison = []
        for perf in report.performance_metrics:
            comparison.append({
                "horizon": perf.horizon,
                "model": perf.model_type,
                "mae": round(perf.mae, 2),
                "rmse": round(perf.rmse, 2),
                "r2": round(perf.r2, 4),
                "samples": perf.samples_trained
            })
        
        return {
            "models": comparison,
            "training_info": report.training_info
        }


# Global singleton
model_health_monitor = ModelHealthMonitor()
