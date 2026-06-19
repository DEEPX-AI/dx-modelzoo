from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from loguru import logger


@dataclass
class ModelEntry:
    """Discovered model metadata from directory scan."""

    name: str
    domain: str
    task: str
    yaml_path: Path
    custom_ops_path: Optional[Path]
    source: Literal["builtin", "custom"] = "builtin"


def discover_models(
    models_dir: Path,
    name: Optional[str] = None,
    domain: Optional[str] = None,
    task: Optional[str] = None,
) -> List[ModelEntry]:
    """Scan models/ directory for YAML config files.

    Directory structure: models/<domain>/<task>/<family_or_model>/<ModelName>.yaml
    Each .yaml file (excluding custom_ops.py etc.) is treated as a model config.
    """
    models_dir = Path(models_dir)
    entries: List[ModelEntry] = []

    for yaml_path in sorted(models_dir.rglob("*.yaml")):
        rel = yaml_path.relative_to(models_dir)
        parts = rel.parts
        # Expect at least: domain/task/folder/file.yaml
        if len(parts) < 4:
            continue

        model_domain = parts[0]
        model_task = parts[1]
        model_dir = yaml_path.parent

        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning("Failed to parse YAML {}: {}", yaml_path, e)
            continue
        if not isinstance(config, dict) or "name" not in config:
            continue
        model_name = config["name"]

        custom_ops = model_dir / "custom_ops.py"
        custom_ops_path = custom_ops if custom_ops.exists() else None

        entry = ModelEntry(
            name=model_name,
            domain=model_domain,
            task=model_task,
            yaml_path=yaml_path,
            custom_ops_path=custom_ops_path,
        )

        if name and entry.name != name:
            continue
        if domain and entry.domain != domain:
            continue
        if task and entry.task != task:
            continue

        entries.append(entry)

    return entries


def scan_all_models(
    models_dir: Path,
    source: Literal["builtin", "custom"] = "builtin",
) -> List[ModelEntry]:
    """Scan models/ directory and return all models with source metadata.

    This is the raw scan behavior that includes source metadata.
    For filtering/resolution logic, use effective_models() or resolve_model().

    Args:
        models_dir: Directory to scan for models
        source: Source type for discovered models (builtin or custom)

    Returns:
        List of ModelEntry objects with source metadata
    """
    models_dir = Path(models_dir)
    entries: List[ModelEntry] = []

    for yaml_path in sorted(models_dir.rglob("*.yaml")):
        rel = yaml_path.relative_to(models_dir)
        parts = rel.parts
        # Expect at least: domain/task/folder/file.yaml
        if len(parts) < 4:
            continue

        model_domain = parts[0]
        model_task = parts[1]
        model_dir = yaml_path.parent

        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning("Failed to parse YAML {}: {}", yaml_path, e)
            continue
        if not isinstance(config, dict) or "name" not in config:
            continue
        model_name = config["name"]

        custom_ops = model_dir / "custom_ops.py"
        custom_ops_path = custom_ops if custom_ops.exists() else None

        entry = ModelEntry(
            name=model_name,
            domain=model_domain,
            task=model_task,
            yaml_path=yaml_path,
            custom_ops_path=custom_ops_path,
            source=source,
        )

        entries.append(entry)

    return entries


def effective_models(all_models: List[ModelEntry]) -> List[ModelEntry]:
    """Return effective models, preferring custom over builtin for same bare name.

    When a custom model has the same name as a builtin model,
    only the custom version is returned.

    Args:
        all_models: Combined list of builtin and custom models

    Returns:
        List of effective models with custom taking precedence
    """
    by_name = {}
    custom_paths_by_name: dict[str, list[Path]] = {}

    for model in all_models:
        if model.source == "custom":
            custom_paths_by_name.setdefault(model.name, []).append(model.yaml_path)

    duplicate_customs = {name: paths for name, paths in custom_paths_by_name.items() if len(paths) > 1}
    if duplicate_customs:
        name, paths = next(iter(duplicate_customs.items()))
        raise ValueError(
            f"Duplicate custom models with name '{name}' found at:\n" + "\n".join(f"  - {path}" for path in paths)
        )

    for model in all_models:
        if model.name not in by_name:
            by_name[model.name] = model
        else:
            # custom takes precedence over builtin
            if model.source == "custom":
                by_name[model.name] = model

    return list(by_name.values())


def resolve_model(
    name: str,
    all_models: List[ModelEntry],
) -> Optional[ModelEntry]:
    """Resolve a model name to a single ModelEntry for info/eval/compile use.

    Resolution rules:
    1. If multiple custom models have the same name, raise error with paths
    2. If custom shadows builtin, warn and return custom
    3. Otherwise return the model (or None if not found)

    Args:
        name: Bare model name to resolve
        all_models: Combined list of builtin and custom models

    Returns:
        Resolved ModelEntry or None if not found

    Raises:
        ValueError: If duplicate custom models with same name exist
    """
    matches = [m for m in all_models if m.name == name]

    if not matches:
        return None

    # Check for duplicate custom names
    custom_matches = [m for m in matches if m.source == "custom"]
    if len(custom_matches) > 1:
        paths = [str(m.yaml_path) for m in custom_matches]
        raise ValueError(
            f"Duplicate custom models with name '{name}' found at:\n" + "\n".join(f"  - {p}" for p in paths)
        )

    # Check if custom shadows builtin
    builtin_matches = [m for m in matches if m.source == "builtin"]
    if custom_matches and builtin_matches:
        logger.warning(f"Custom model '{name}' shadows builtin model at {builtin_matches[0].yaml_path}")

    # Return custom if available, otherwise builtin
    if custom_matches:
        return custom_matches[0]
    return builtin_matches[0] if builtin_matches else None
