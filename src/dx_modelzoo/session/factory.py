"""Session factory — creates inference sessions from paths or profile names.

Resolution order:
1. If model_or_profile is a file path (.onnx/.dxnn) and exists → create session directly
2. If model_or_profile is a profile name → resolve model path from YAML artifacts
   a. File exists locally → create session
   b. File missing → attempt download from DXMZ_MODEL_URL
   c. Download fails → raise SessionCreationError
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from dx_modelzoo.session import SessionBase
    from dx_modelzoo.session.runtime_config import RuntimeConfig

# Download base URL — set via environment variable
# Example: DXMZ_MODEL_URL=https://models.deepx.ai/v1
#   Full URL: {DXMZ_MODEL_URL}/{model_name}/{filename}
DXMZ_MODEL_URL_ENV = "DXMZ_MODEL_URL"


class SessionCreationError(Exception):
    """Raised when a session cannot be created."""


def _detect_target(path: str) -> str:
    """Detect session type from file extension."""
    p = Path(path)
    if p.suffix == ".onnx":
        return "onnx"
    elif p.suffix == ".dxnn":
        return "dxnn"
    raise SessionCreationError(
        f"Cannot determine session type from extension: '{p.suffix}'. " "Expected .onnx or .dxnn"
    )


def _create_session(
    path: str, target: str, runtime_config: Optional["RuntimeConfig"] = None
) -> "SessionBase":
    """Create a session instance from a resolved file path."""
    if target == "dxnn":
        from dx_modelzoo.session.dx_session import DxRuntimeSession

        return DxRuntimeSession(path, runtime_config)
    else:
        from dx_modelzoo.session.onnx_session import OnnxRuntimeSession

        return OnnxRuntimeSession(path, runtime_config)


def _download_model(url: str, dest: Path) -> None:
    """Download a model file from URL to local path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model: {url} → {dest}")
    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.success(f"Download complete: {dest}")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise SessionCreationError(
            f"Failed to download model from {url}: {e}\n"
            f"Set {DXMZ_MODEL_URL_ENV} environment variable to the correct base URL, "
            "or provide the model file manually."
        ) from e


def _try_download(model_name: str, compile_profile: str, dest_path: Path) -> Path:
    """Attempt to download a model file if DXMZ_MODEL_URL is set."""
    base_url = os.environ.get(DXMZ_MODEL_URL_ENV, "").rstrip("/")
    if not base_url:
        raise SessionCreationError(
            f"Model file not found: {dest_path}\n"
            f"To enable auto-download, set {DXMZ_MODEL_URL_ENV} environment variable.\n"
            f"  export {DXMZ_MODEL_URL_ENV}=https://your-model-server.com/models"
        )

    suffix = Path(dest_path).suffix
    branch = "" if suffix == ".onnx" else f"latest/{compile_profile}/"
    url = f"{base_url}/{suffix[1:]}/{branch}{model_name}{suffix}"
    _download_model(url, dest_path)
    return dest_path


def create_session(
    model_or_profile: str,
    *,
    builder: Optional[object] = None,
    runtime_config: Optional["RuntimeConfig"] = None,
) -> "SessionBase":
    """Create an inference session.

    Args:
        model_or_profile: Either a file path (.onnx/.dxnn) or a profile name.
            - File path: creates session directly if file exists.
            - Profile name: requires `builder` to resolve model path from YAML.
        builder: ModelBuilder instance (required when model_or_profile is a profile name).
        runtime_config: Runtime options (device, batch_size, async, ...) passed
            to the session. When ``None`` and a profile name is given, it is
            derived from the profile's ``runtime`` block.

    Returns:
        SessionBase instance.

    Raises:
        SessionCreationError: If session cannot be created.
    """
    path = Path(model_or_profile)

    # Case 1: Direct file path
    if path.suffix in (".onnx", ".dxnn") and path.exists():
        target = _detect_target(str(path))
        logger.info(f"Creating {target} session from: {path}")
        return _create_session(str(path), target, runtime_config)

    # Case 2: Profile name — requires builder
    if builder is None:
        # Could be a path that doesn't exist
        if path.suffix in (".onnx", ".dxnn"):
            raise SessionCreationError(f"Model file not found: {path}")
        raise SessionCreationError(
            f"'{model_or_profile}' is not a file path. " "Provide a ModelBuilder to resolve profile names."
        )

    return _create_session_from_profile(model_or_profile, builder, runtime_config)


def _create_session_from_profile(
    profile_name: str, builder: object, runtime_config: Optional["RuntimeConfig"] = None
) -> "SessionBase":
    """Resolve model path from YAML profile and create session."""
    from dx_modelzoo.session.runtime_config import RuntimeConfig

    profile = builder.get_profile(profile_name)
    target = profile.get("target")

    if target is None:
        target = "onnx" if profile_name == "onnx" else "dxnn"
        logger.warning(f"No target specified for profile '{profile_name}'. " f"Defaulting to '{target}'.")

    if runtime_config is None:
        runtime_config = RuntimeConfig.from_profile(profile, target=target)

    # Resolve model path from artifacts
    artifacts = builder.config.get("artifacts", {})
    if not artifacts:
        raise SessionCreationError(f"No artifacts in {builder.yaml_path} for profile '{profile_name}'")

    base_path = artifacts.get("path") or artifacts.get(target)
    if base_path is None:
        raise SessionCreationError(f"No artifact path in {builder.yaml_path}")

    base_path = Path(base_path)
    model_name = builder.config["name"]

    if target == "onnx":
        filename = f"{model_name}.onnx"
        model_path = base_path if base_path.suffix == ".onnx" else base_path / filename
    else:  # dxnn
        filename = f"{model_name}_{profile_name}.dxnn"
        dxnn_dir = base_path if base_path.is_dir() or not base_path.suffix else base_path.parent
        model_path = dxnn_dir / filename

    # File exists — create session
    if model_path.exists():
        return _create_session(str(model_path), target, runtime_config)

    # File missing — try download
    logger.warning(f"Model file not found locally: {model_path}")
    _try_download(model_name, profile_name, model_path)
    return _create_session(str(model_path), target, runtime_config)
