"""
Data Quality Monitoring & Health Check Module.

Provides real-time data quality metrics for:
- Feature Store freshness
- Data completeness
- Value validity
- Temporal coverage
- Sensor availability

Used by Dashboard and Flask API to expose actual data health status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "model_features_3cities.csv"

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    
    status: str  # "healthy", "warning", "critical", "unavailable"
    overall_score: int  # 0-100
    last_update: Optional[str]
    freshness_hours: Optional[float]
    checks: Dict[str, Dict[str, Any]]
    issues: List[str]
    metrics: Dict[str, Any]


class DataQualityMonitor:
    """Real-time data quality monitoring for Feature Store."""
    
    def __init__(self, feature_file: Path = FEATURE_FILE):
        self.feature_file = feature_file
        
    def get_quality_report(self, city: Optional[str] = None) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Args:
            city: Optional city filter (Lahore, Islamabad, Faisalabad)
            
        Returns:
            DataQualityReport with real health status
        """
        if not self.feature_file.exists():
            return DataQualityReport(
                status="unavailable",
                overall_score=0,
                last_update=None,
                freshness_hours=None,
                checks={},
                issues=["Feature store file not found"],
                metrics={}
            )
        
        try:
            df = pd.read_csv(self.feature_file)
            
            if city:
                df = df[df["city"].str.lower() == city.lower()].copy()
            
            if df.empty:
                return DataQualityReport(
                    status="unavailable",
                    overall_score=0,
                    last_update=None,
                    freshness_hours=None,
                    checks={},
                    issues=[f"No data available{f' for {city}' if city else ''}"],
                    metrics={}
                )
            
            # Parse timestamp
            if "hour" in df.columns:
                df["timestamp"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            else:
                df["timestamp"] = pd.NaT
            
            # Run quality checks
            checks = {}
            issues = []
            
            # 1. Freshness Check
            freshness_check = self._check_freshness(df)
            checks["freshness"] = freshness_check
            if freshness_check["status"] != "pass":
                issues.append(freshness_check["message"])
            
            # 2. Completeness Check
            completeness_check = self._check_completeness(df)
            checks["completeness"] = completeness_check
            if completeness_check["status"] != "pass":
                issues.append(completeness_check["message"])
            
            # 3. Validity Check
            validity_check = self._check_validity(df)
            checks["validity"] = validity_check
            if validity_check["status"] != "pass":
                issues.append(validity_check["message"])
            
            # 4. Coverage Check
            coverage_check = self._check_coverage(df)
            checks["coverage"] = coverage_check
            if coverage_check["status"] != "pass":
                issues.append(coverage_check["message"])
            
            # Calculate overall score and status
            score = self._calculate_score(checks)
            status = self._determine_status(score, issues)
            
            # Get latest timestamp
            last_update = None
            freshness_hours = None
            if not df["timestamp"].isna().all():
                latest = df["timestamp"].max()
                if pd.notna(latest):
                    last_update = latest.isoformat()
                    now = datetime.now(timezone.utc)
                    freshness_hours = (now - latest).total_seconds() / 3600
            
            # Collect metrics
            metrics = {
                "total_records": int(len(df)),
                "cities": int(df["city"].nunique()) if "city" in df.columns else 0,
                "date_range_days": self._get_date_range_days(df),
                "feature_count": len([c for c in df.columns if c not in {"city", "hour", "timestamp"}]),
                "null_percentage": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            }
            
            return DataQualityReport(
                status=status,
                overall_score=score,
                last_update=last_update,
                freshness_hours=freshness_hours,
                checks=checks,
                issues=issues,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return DataQualityReport(
                status="critical",
                overall_score=0,
                last_update=None,
                freshness_hours=None,
                checks={},
                issues=[f"Quality check error: {str(e)}"],
                metrics={}
            )
    
    def _check_freshness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data freshness (< 48 hours = pass, < 7 days = warning)."""
        if "timestamp" not in df.columns or df["timestamp"].isna().all():
            return {
                "name": "Data Freshness",
                "status": "fail",
                "message": "Timestamp data unavailable",
                "score": 0
            }
        
        latest = df["timestamp"].max()
        if pd.isna(latest):
            return {
                "name": "Data Freshness",
                "status": "fail",
                "message": "No valid timestamps",
                "score": 0
            }
        
        now = datetime.now(timezone.utc)
        hours_old = (now - latest).total_seconds() / 3600
        
        if hours_old < 48:
            return {
                "name": "Data Freshness",
                "status": "pass",
                "message": f"Data is fresh ({hours_old:.1f} hours old)",
                "score": 100,
                "hours_old": round(hours_old, 1)
            }
        elif hours_old < 168:  # 7 days
            return {
                "name": "Data Freshness",
                "status": "warning",
                "message": f"Data is stale ({hours_old:.1f} hours old)",
                "score": 60,
                "hours_old": round(hours_old, 1)
            }
        else:
            return {
                "name": "Data Freshness",
                "status": "fail",
                "message": f"Data is very stale ({hours_old:.1f} hours old)",
                "score": 20,
                "hours_old": round(hours_old, 1)
            }
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness (missing values)."""
        critical_cols = ["pm2_5", "temperature", "humidity", "pressure"]
        present_critical = [c for c in critical_cols if c in df.columns]
        
        if not present_critical:
            return {
                "name": "Data Completeness",
                "status": "fail",
                "message": "Critical features missing",
                "score": 0
            }
        
        # Calculate null percentage for critical columns
        null_pct = df[present_critical].isna().sum().sum() / (len(df) * len(present_critical)) * 100
        
        if null_pct < 10:
            return {
                "name": "Data Completeness",
                "status": "pass",
                "message": f"Data is complete ({null_pct:.1f}% missing)",
                "score": 100,
                "null_percentage": round(null_pct, 1)
            }
        elif null_pct < 30:
            return {
                "name": "Data Completeness",
                "status": "warning",
                "message": f"Moderate missing data ({null_pct:.1f}% missing)",
                "score": 60,
                "null_percentage": round(null_pct, 1)
            }
        else:
            return {
                "name": "Data Completeness",
                "status": "fail",
                "message": f"High missing data ({null_pct:.1f}% missing)",
                "score": 20,
                "null_percentage": round(null_pct, 1)
            }
    
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check value validity (no NaN, Inf, negative pollutants)."""
        issues = []
        
        # Check PM2.5
        if "pm2_5" in df.columns:
            if (df["pm2_5"] < 0).any():
                issues.append("Negative PM2.5 values detected")
            if np.isinf(df["pm2_5"]).any():
                issues.append("Infinite PM2.5 values detected")
        
        # Check temperature range
        if "temperature" in df.columns:
            if ((df["temperature"] < -50) | (df["temperature"] > 60)).any():
                issues.append("Implausible temperature values")
        
        # Check humidity range
        if "humidity" in df.columns:
            if ((df["humidity"] < 0) | (df["humidity"] > 100)).any():
                issues.append("Invalid humidity values")
        
        if not issues:
            return {
                "name": "Data Validity",
                "status": "pass",
                "message": "All values within valid ranges",
                "score": 100
            }
        else:
            return {
                "name": "Data Validity",
                "status": "warning",
                "message": "; ".join(issues),
                "score": 50
            }
    
    def _check_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal coverage (gaps in time series)."""
        if "timestamp" not in df.columns or df["timestamp"].isna().all():
            return {
                "name": "Temporal Coverage",
                "status": "fail",
                "message": "No temporal data",
                "score": 0
            }
        
        timestamps = df["timestamp"].dropna().sort_values()
        if len(timestamps) < 2:
            return {
                "name": "Temporal Coverage",
                "status": "warning",
                "message": "Insufficient temporal data",
                "score": 40
            }
        
        # Check for large gaps (> 24 hours)
        time_diffs = timestamps.diff().dropna()
        max_gap_hours = time_diffs.max().total_seconds() / 3600
        
        if max_gap_hours < 24:
            return {
                "name": "Temporal Coverage",
                "status": "pass",
                "message": f"Continuous coverage (max gap: {max_gap_hours:.1f}h)",
                "score": 100,
                "max_gap_hours": round(max_gap_hours, 1)
            }
        elif max_gap_hours < 72:
            return {
                "name": "Temporal Coverage",
                "status": "warning",
                "message": f"Moderate gaps detected (max gap: {max_gap_hours:.1f}h)",
                "score": 60,
                "max_gap_hours": round(max_gap_hours, 1)
            }
        else:
            return {
                "name": "Temporal Coverage",
                "status": "fail",
                "message": f"Large gaps detected (max gap: {max_gap_hours:.1f}h)",
                "score": 30,
                "max_gap_hours": round(max_gap_hours, 1)
            }
    
    def _calculate_score(self, checks: Dict[str, Dict[str, Any]]) -> int:
        """Calculate overall quality score (0-100)."""
        if not checks:
            return 0
        
        scores = [check["score"] for check in checks.values()]
        return int(sum(scores) / len(scores))
    
    def _determine_status(self, score: int, issues: List[str]) -> str:
        """Determine overall status based on score and issues."""
        if score >= 80 and not issues:
            return "healthy"
        elif score >= 60:
            return "warning"
        elif score > 0:
            return "critical"
        else:
            return "unavailable"
    
    def _get_date_range_days(self, df: pd.DataFrame) -> Optional[int]:
        """Get date range in days."""
        if "timestamp" not in df.columns or df["timestamp"].isna().all():
            return None
        
        timestamps = df["timestamp"].dropna()
        if len(timestamps) < 2:
            return None
        
        date_range = (timestamps.max() - timestamps.min()).total_seconds() / 86400
        return int(date_range)


# Global singleton
data_quality_monitor = DataQualityMonitor()
