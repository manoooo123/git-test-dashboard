"""
Model Registry & Versioning System.

Provides comprehensive model lifecycle management:
- Model registration with metadata
- Version tracking
- Performance comparison
- Deployment tracking
- Artifact integrity verification
- Rollback support

Used for production model management and audit trails.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
REGISTRY_FILE = PROJECT_ROOT / "models" / "model_registry.json"

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model registration metadata."""
    
    model_id: str
    horizon: str
    algorithm: str
    version: str
    created_at: str
    file_path: str
    file_size_mb: float
    checksum: str
    performance: Dict[str, float]
    training_samples: int
    feature_count: int
    feature_columns: List[str]
    deployment_status: str  # "production", "staging", "archived"
    notes: str


class ModelRegistry:
    """
    Model Registry for tracking trained model artifacts.
    
    Maintains a JSON-based registry of all trained models with:
    - Version history
    - Performance metrics
    - Deployment status
    - Integrity checksums
    """
    
    def __init__(self, registry_file: Path = REGISTRY_FILE):
        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk."""
        if not self.registry_file.exists():
            return {"models": [], "last_updated": None}
        
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            return {"models": [], "last_updated": None}
    
    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save registry to disk."""
        try:
            registry["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, default=str)
            logger.info(f"Model registry saved: {self.registry_file}")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of model file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return "unknown"
    
    def register_model(
        self,
        horizon: str,
        algorithm: str,
        version: str,
        file_path: Path,
        performance: Dict[str, float],
        training_samples: int,
        feature_count: int,
        feature_columns: List[str],
        deployment_status: str = "staging",
        notes: str = ""
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            horizon: Forecast horizon (24h, 48h, 72h)
            algorithm: Model algorithm (Ridge, RandomForest, DeepLearning)
            version: Model version (e.g., v2.0.0)
            file_path: Path to model artifact
            performance: Dict with MAE, RMSE, R2
            training_samples: Number of training samples
            feature_count: Number of features
            feature_columns: List of feature names
            deployment_status: production, staging, or archived
            notes: Optional notes
            
        Returns:
            model_id: Unique model identifier
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {file_path}")
        
        # Generate model ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_id = f"{algorithm}_{horizon}_{timestamp}"
        
        # Calculate metadata
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        checksum = self._calculate_checksum(file_path)
        created_at = datetime.now(timezone.utc).isoformat()
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            horizon=horizon,
            algorithm=algorithm,
            version=version,
            created_at=created_at,
            file_path=str(file_path),
            file_size_mb=round(file_size_mb, 2),
            checksum=checksum,
            performance=performance,
            training_samples=training_samples,
            feature_count=feature_count,
            feature_columns=feature_columns,
            deployment_status=deployment_status,
            notes=notes
        )
        
        # Load and update registry
        registry = self._load_registry()
        registry["models"].append(asdict(metadata))
        self._save_registry(registry)
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def get_production_model(self, horizon: str) -> Optional[ModelMetadata]:
        """Get currently deployed production model for horizon."""
        registry = self._load_registry()
        
        for model_data in reversed(registry["models"]):
            if (model_data["horizon"] == horizon and 
                model_data["deployment_status"] == "production"):
                return ModelMetadata(**model_data)
        
        return None
    
    def get_all_models(self, horizon: Optional[str] = None) -> List[ModelMetadata]:
        """Get all registered models, optionally filtered by horizon."""
        registry = self._load_registry()
        models = []
        
        for model_data in registry["models"]:
            if horizon is None or model_data["horizon"] == horizon:
                models.append(ModelMetadata(**model_data))
        
        return models
    
    def promote_to_production(self, model_id: str) -> bool:
        """
        Promote a model to production status.
        Demotes any existing production model for the same horizon.
        
        Args:
            model_id: Model identifier to promote
            
        Returns:
            True if successful
        """
        registry = self._load_registry()
        target_horizon = None
        target_found = False
        
        # Find target model and its horizon
        for model in registry["models"]:
            if model["model_id"] == model_id:
                target_horizon = model["horizon"]
                target_found = True
                break
        
        if not target_found:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        # Demote existing production models for this horizon
        for model in registry["models"]:
            if (model["horizon"] == target_horizon and 
                model["deployment_status"] == "production"):
                model["deployment_status"] = "archived"
                logger.info(f"Archived previous production model: {model['model_id']}")
        
        # Promote target model
        for model in registry["models"]:
            if model["model_id"] == model_id:
                model["deployment_status"] = "production"
                logger.info(f"Promoted to production: {model_id}")
        
        self._save_registry(registry)
        return True
    
    def verify_integrity(self, model_id: str) -> bool:
        """
        Verify model artifact integrity using checksum.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if checksum matches
        """
        registry = self._load_registry()
        
        for model in registry["models"]:
            if model["model_id"] == model_id:
                file_path = Path(model["file_path"])
                stored_checksum = model["checksum"]
                
                if not file_path.exists():
                    logger.error(f"Model artifact not found: {file_path}")
                    return False
                
                current_checksum = self._calculate_checksum(file_path)
                
                if current_checksum != stored_checksum:
                    logger.error(f"Checksum mismatch for {model_id}")
                    return False
                
                logger.info(f"Integrity verified: {model_id}")
                return True
        
        logger.error(f"Model {model_id} not found in registry")
        return False
    
    def get_performance_comparison(self, horizon: str) -> List[Dict[str, Any]]:
        """
        Compare performance across all models for a horizon.
        
        Args:
            horizon: Forecast horizon (24h, 48h, 72h)
            
        Returns:
            List of model performance comparisons
        """
        models = self.get_all_models(horizon=horizon)
        
        comparison = []
        for model in models:
            comparison.append({
                "model_id": model.model_id,
                "algorithm": model.algorithm,
                "version": model.version,
                "status": model.deployment_status,
                "MAE": model.performance.get("MAE", 0),
                "RMSE": model.performance.get("RMSE", 0),
                "R2": model.performance.get("R2", 0),
                "created_at": model.created_at,
            })
        
        # Sort by R² descending
        comparison.sort(key=lambda x: x["R2"], reverse=True)
        return comparison
    
    def export_registry_report(self) -> Dict[str, Any]:
        """Export comprehensive registry report for monitoring."""
        registry = self._load_registry()
        
        total_models = len(registry["models"])
        production_models = sum(
            1 for m in registry["models"] 
            if m["deployment_status"] == "production"
        )
        
        horizons = ["24h", "48h", "72h"]
        horizon_summary = {}
        
        for horizon in horizons:
            h_models = [m for m in registry["models"] if m["horizon"] == horizon]
            prod_model = next(
                (m for m in reversed(h_models) if m["deployment_status"] == "production"),
                None
            )
            
            horizon_summary[horizon] = {
                "total_versions": len(h_models),
                "production_model": prod_model["model_id"] if prod_model else None,
                "production_performance": prod_model["performance"] if prod_model else None,
            }
        
        return {
            "total_models": total_models,
            "production_models": production_models,
            "last_updated": registry["last_updated"],
            "horizon_summary": horizon_summary,
        }


# Global singleton
model_registry = ModelRegistry()
