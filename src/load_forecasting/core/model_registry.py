"""
Save, load, and list trained models.

Provides:
- Persistent storage with versioning
- Model metadata tracking
- Scaler persistence for inverse transforms
- Version compatibility checking
"""

import json
import pickle
import importlib.metadata
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any
import logging
import os

from darts.models import RegressionModel, NaiveMean, NaiveSeasonal, NaiveMovingAverage

logger = logging.getLogger(__name__)

# Version strings are constant for the lifetime of the process — cache them once.
try:
    _DARTS_VERSION: str = importlib.metadata.version("darts")
except Exception:
    _DARTS_VERSION = "unknown"

try:
    _PKG_VERSION: str = importlib.metadata.version("load-forecasting-mcp")
except Exception:
    _PKG_VERSION = "0.1.0"

_PYTHON_VERSION: str = (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)


class ModelNotFoundError(Exception):
    """Raised when a model is not found in the registry."""

    pass


class ModelIncompatibleError(Exception):
    """Raised when a model was trained with an incompatible version."""

    pass


class ModelCorruptedError(Exception):
    """Raised when a model file cannot be loaded."""

    pass


class ModelRegistry:
    """Registry for trained forecasting models."""

    # In-memory cache keyed by registry file path so that multiple registries
    # pointing to different base_dirs don't share state (important for tests).
    _registry_cache: dict[str, dict] = {}

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize model registry.

        Args:
            base_dir: Base directory for model storage.
                     Defaults to LOAD_FORECASTING_MODEL_DIR env var or "./models"
        """
        if base_dir is None:
            base_dir = os.getenv("LOAD_FORECASTING_MODEL_DIR", "models")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "registry.json"

        # Initialize registry if not exists
        if not self.registry_path.exists():
            self._init_registry()

    def _init_registry(self) -> None:
        """Initialize empty registry file."""
        registry = {
            "version": "1.0",
            "models": [],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._save_registry(registry)

    def _load_registry(self) -> dict:
        """Load registry from disk (or return in-memory cache for this path)."""
        cache_key = str(self.registry_path)
        if cache_key in ModelRegistry._registry_cache:
            return ModelRegistry._registry_cache[cache_key]

        if not self.registry_path.exists():
            self._init_registry()
            return ModelRegistry._registry_cache[cache_key]

        with open(self.registry_path, "r", encoding="utf-8") as f:
            ModelRegistry._registry_cache[cache_key] = json.load(f)

        return ModelRegistry._registry_cache[cache_key]

    def _save_registry(self, registry: dict) -> None:
        """Save registry to disk and update in-memory cache."""
        registry["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)
        # Keep cache consistent with what was just written
        ModelRegistry._registry_cache[str(self.registry_path)] = registry

    def generate_model_id(
        self,
        building_name: Optional[str],
        model_type: str,
    ) -> str:
        """
        Generate unique model ID.

        Format: {building}_{model_type}_{YYYYMMDD}_{HHMMSS}

        Args:
            building_name: Building identifier
            model_type: Type of model

        Returns:
            Unique model ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        building = (
            building_name.replace(" ", "_").replace("/", "_")
            if building_name
            else "unnamed"
        )
        return f"{building}_{model_type}_{timestamp}"

    def save_model(
        self,
        model: Any,
        model_id: str,
        model_type: str,
        building_name: Optional[str],
        config: dict,
        column_mapping: dict,
        data_info: dict,
        training_metrics: dict,
        validation_metrics: dict,
        scalers: Optional[dict] = None,
    ) -> str:
        """
        Save model and metadata to registry.

        Args:
            model: Trained Darts model
            model_id: Unique model identifier
            model_type: Type of model (e.g., "LinearRegression")
            building_name: Building identifier
            config: Training configuration
            column_mapping: Column mapping used
            data_info: Data summary
            training_metrics: Training set metrics
            validation_metrics: Validation set metrics
            scalers: Dict with target_scaler and covariate_scaler

        Returns:
            Path to saved model directory
        """
        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Build metadata
        metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "darts_class": f"darts.models.{type(model).__name__}",
            "building_name": building_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "column_mapping": column_mapping,
            "data_info": data_info,
            "metrics": {
                "training": training_metrics,
                "validation": validation_metrics,
            },
            "version_info": {
                "darts_version": _DARTS_VERSION,
                "load_forecasting_version": _PKG_VERSION,
                "python_version": _PYTHON_VERSION,
            },
        }

        # Save model
        model_path = model_dir / "model.pkl"
        try:
            model.save(str(model_path))
        except Exception:
            # Fallback to pickle if Darts save fails
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save scalers if provided
        if scalers:
            scalers_dir = model_dir / "scalers"
            scalers_dir.mkdir(exist_ok=True)

            if "target_scaler" in scalers and scalers["target_scaler"] is not None:
                with open(scalers_dir / "target_scaler.pkl", "wb") as f:
                    pickle.dump(scalers["target_scaler"], f)

            if "covariate_scaler" in scalers and scalers["covariate_scaler"] is not None:
                with open(scalers_dir / "covariate_scaler.pkl", "wb") as f:
                    pickle.dump(scalers["covariate_scaler"], f)

            if "future_covariate_scaler" in scalers and scalers["future_covariate_scaler"] is not None:
                with open(scalers_dir / "future_covariate_scaler.pkl", "wb") as f:
                    pickle.dump(scalers["future_covariate_scaler"], f)

        # Update registry index
        self._update_registry(model_id, model_type, building_name, validation_metrics, config)

        logger.info(f"Model saved: {model_id}")
        return str(model_dir)

    def _update_registry(
        self,
        model_id: str,
        model_type: str,
        building_name: Optional[str],
        validation_metrics: dict,
        config: dict,
    ) -> None:
        """Update registry index with new model."""
        registry = self._load_registry()

        # Remove existing entry if present (update case)
        registry["models"] = [m for m in registry["models"] if m["model_id"] != model_id]

        # Add new entry
        entry = {
            "model_id": model_id,
            "model_type": model_type,
            "building_name": building_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "validation_cv_rmse": validation_metrics.get("cv_rmse"),
            "lookback_hours": config.get("lookback_hours"),
            "horizon_hours": config.get("horizon_hours"),
            "path": f"models/{model_id}",
        }
        registry["models"].append(entry)

        self._save_registry(registry)

    def load_model(self, model_id: str) -> tuple[Any, dict, dict]:
        """
        Load model and metadata from registry.

        Args:
            model_id: Unique model identifier

        Returns:
            Tuple of (model, metadata, scalers)

        Raises:
            ModelNotFoundError: If model_id doesn't exist
            ModelCorruptedError: If model file cannot be loaded
        """
        model_dir = self.base_dir / model_id

        if not model_dir.exists():
            raise ModelNotFoundError(f"Model not found: {model_id}")

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise ModelCorruptedError(f"Metadata not found for model: {model_id}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load model
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise ModelCorruptedError(f"Model file not found: {model_id}")

        try:
            # Try Darts load first (models already imported at module level)
            model_type = metadata.get("model_type", "")
            if model_type == "LinearRegression":
                model = RegressionModel.load(str(model_path))
            elif model_type == "NaiveMean":
                model = NaiveMean.load(str(model_path))
            elif model_type == "NaiveSeasonal":
                model = NaiveSeasonal.load(str(model_path))
            elif model_type == "NaiveMovingAverage":
                model = NaiveMovingAverage.load(str(model_path))
            else:
                # Fallback to pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
        except Exception as e:
            # Fallback to pickle
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            except Exception as e2:
                raise ModelCorruptedError(f"Failed to load model: {e2}")

        # Load scalers
        scalers = {}
        scalers_dir = model_dir / "scalers"
        if scalers_dir.exists():
            target_scaler_path = scalers_dir / "target_scaler.pkl"
            if target_scaler_path.exists():
                with open(target_scaler_path, "rb") as f:
                    scalers["target_scaler"] = pickle.load(f)

            covariate_scaler_path = scalers_dir / "covariate_scaler.pkl"
            if covariate_scaler_path.exists():
                with open(covariate_scaler_path, "rb") as f:
                    scalers["covariate_scaler"] = pickle.load(f)

            future_covariate_scaler_path = scalers_dir / "future_covariate_scaler.pkl"
            if future_covariate_scaler_path.exists():
                with open(future_covariate_scaler_path, "rb") as f:
                    scalers["future_covariate_scaler"] = pickle.load(f)

        logger.info(f"Model loaded: {model_id}")
        return model, metadata, scalers

    def list_models(
        self,
        building_name: Optional[str] = None,
        model_type: Optional[str] = None,
        sort_by: str = "created_at",
        limit: int = 20,
    ) -> tuple[list[dict], int]:
        """
        List models in registry with optional filters.

        Args:
            building_name: Filter by building
            model_type: Filter by model type
            sort_by: Sort field (created_at, validation_cv_rmse, model_type)
            limit: Maximum results

        Returns:
            Tuple of (filtered_models_list, total_count)
        """
        registry = self._load_registry()
        models = registry.get("models", [])

        # Apply filters
        if building_name:
            models = [m for m in models if m.get("building_name") == building_name]

        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]

        total_count = len(models)

        # Sort
        reverse = sort_by == "created_at"  # Newest first for created_at
        if sort_by == "validation_cv_rmse":
            # Lower is better, so ascending
            models = sorted(
                models,
                key=lambda m: m.get("validation_cv_rmse") or float("inf"),
            )
        elif sort_by == "model_type":
            models = sorted(models, key=lambda m: m.get("model_type", ""))
        else:  # created_at
            models = sorted(
                models,
                key=lambda m: m.get("created_at", ""),
                reverse=reverse,
            )

        # Apply limit
        models = models[:limit]

        return models, total_count

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from registry.

        Args:
            model_id: Model to delete

        Returns:
            True if deleted, False if not found
        """
        model_dir = self.base_dir / model_id

        if not model_dir.exists():
            return False

        # Remove directory
        shutil.rmtree(model_dir)

        # Update registry
        registry = self._load_registry()
        registry["models"] = [m for m in registry["models"] if m["model_id"] != model_id]
        self._save_registry(registry)

        logger.info(f"Model deleted: {model_id}")
        return True
