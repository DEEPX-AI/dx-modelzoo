"""Integration test conftest — shared fixtures for integration tests."""

import pytest
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent.parent / "src" / "dx_modelzoo" / "models"


@pytest.fixture
def models_dir():
    """Path to builtin models directory."""
    return MODELS_DIR


@pytest.fixture
def all_yaml_paths(models_dir):
    """Collect all YAML config files under src/dx_modelzoo/models/cv/."""
    cv_dir = models_dir / "cv"
    if not cv_dir.exists():
        pytest.skip("models/cv directory not found")
    return sorted(cv_dir.rglob("*.yaml"))
