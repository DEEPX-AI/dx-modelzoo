"""Path utilities for dx-modelzoo.

This module provides shared path resolution functions to avoid
duplication across CLI, commands, and TUI modules.
"""

from pathlib import Path


def get_builtin_models_dir() -> Path:
    """Get the builtin models directory path.

    Returns the absolute path to the builtin models directory
    within the dx-modelzoo package installation.

    Returns:
        Path to the builtin models directory
    """
    from dx_modelzoo import __file__ as package_init

    return Path(package_init).parent / "models"


def get_workspace_custom_dir() -> Path:
    """Get the workspace-local custom models directory.

    Returns the path to the 'custom' directory in the current
    working directory, where user-defined custom models are stored.

    Returns:
        Path to the workspace custom directory
    """
    return Path.cwd() / "custom"
