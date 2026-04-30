"""Tests for model_registry module."""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.model_registry import (
    ModelRegistry,
    ModelNotFoundError,
)


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_dir=tmpdir)
        yield registry


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_generate_model_id(self, temp_registry):
        """Test model ID generation."""
        model_id = temp_registry.generate_model_id("Building_33", "LinearRegression")

        assert "Building_33" in model_id
        assert "LinearRegression" in model_id
        assert len(model_id.split("_")) >= 4  # building_model_date_time

    def test_generate_model_id_no_building(self, temp_registry):
        """Test model ID generation without building name."""
        model_id = temp_registry.generate_model_id(None, "NaiveMean")

        assert "unnamed" in model_id
        assert "NaiveMean" in model_id

    def test_list_models_empty(self, temp_registry):
        """Test listing models from empty registry."""
        models, count = temp_registry.list_models()

        assert models == []
        assert count == 0

    def test_model_not_found(self, temp_registry):
        """Test loading non-existent model."""
        with pytest.raises(ModelNotFoundError):
            temp_registry.load_model("nonexistent_model_id")

    def test_init_registry(self, temp_registry):
        """Test registry initialization."""
        assert temp_registry.registry_path.exists()

        registry = temp_registry._load_registry()
        assert "version" in registry
        assert "models" in registry
        assert registry["models"] == []
