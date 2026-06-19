from __future__ import annotations

import os
import re
from typing import Any

from loguru import logger


def resolve_variables(value: str, yaml_path: str, extra_vars: dict = None) -> str:
    """Replace ${VAR} patterns with environment variable values or extra_vars.

    Lookup order: extra_vars > environment variables.
    If a variable is not set, it is left as-is (e.g. '${MODEL_ROOT}').
    """

    def replacer(match):
        var_name = match.group(1)
        if extra_vars and var_name in extra_vars:
            return extra_vars[var_name]
        val = os.environ.get(var_name)
        if val is None:
            logger.warning("Unresolved variable ${{{}}}: not set in environment", var_name)
            return match.group(0)  # keep original ${VAR}
        return val

    return re.sub(r"\$\{(\w+)\}", replacer, value)


def resolve_variables_recursive(data: Any, yaml_path: str, extra_vars: dict = None) -> Any:
    """Recursively resolve ${VAR} in all string values of a nested structure."""
    if isinstance(data, str):
        return resolve_variables(data, yaml_path, extra_vars)
    elif isinstance(data, dict):
        # Extract MODEL_NAME from top-level 'name' field before resolving
        if extra_vars is None:
            extra_vars = {}
        if "name" in data and isinstance(data["name"], str):
            extra_vars = {**extra_vars, "MODEL_NAME": data["name"]}
        return {k: resolve_variables_recursive(v, yaml_path, extra_vars) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_variables_recursive(item, yaml_path, extra_vars) for item in data]
    return data
