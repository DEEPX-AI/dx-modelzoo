"""Create wizard for dxmz create command.

This module implements a lightweight terminal wizard for creating new model configurations.
The wizard collects all necessary parameters through sequential prompts and
produces a CreateWizardResult that can be used to generate model YAML files.

Architecture:
- Helper functions (get_supported_*, normalize_*, build_*) provide data and transformations
- CreateWizardResult dataclass holds the final collected data
- Prompt helpers (prompt_select, prompt_text, prompt_multiselect) provide interactive input
- run_create_wizard() provides the main entry point for CLI integration

The wizard implements supported-first selection:
- Domain/task choices come from builtin models tree (models/<domain>/<task>)
- Dataset choices come from DATASET_REGISTRY
- Evaluator names come from EVALUATOR_REGISTRY
- Custom fallback is available for task and dataset when "[Custom]" is selected

The implementation uses sequential prompts that feel lightweight and fast, similar to
Vite or other modern CLI tools, while preserving all the validation and data collection
capabilities of the original wizard.
"""

from __future__ import annotations

import inspect
import math
import re
import shutil
import sys
import termios
import tty
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt

from dx_modelzoo.dataset import DATASET_REGISTRY
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY
from dx_modelzoo.loader.model_scaffold import IDENTIFIER_PATTERN, parse_input_shape, validate_identifier
from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY

__all__ = [
    "BACK",
    "CreateWizardResult",
    "run_create_wizard",
    "get_supported_domains",
    "get_supported_tasks",
    "get_supported_datasets",
    "get_supported_evaluators",
    "get_preprocessing_options",
    "get_postprocessing_options",
    "normalize_family",
    "validate_custom_task",
    "validate_custom_dataset",
    "validate_input_shape",
    "build_preprocessing_steps",
    "build_postprocessing_steps",
    "result_to_dict",
    "format_step_as_function",
    "format_step_header",
    "format_completion_summary",
    "get_datasets_for_task",
    "get_preprocessing_for_task",
    "get_postprocessing_for_task",
]


@dataclass
class CreateWizardResult:
    """Result payload from the create wizard."""

    domain: str
    task: str
    dataset_name: str
    family: str | None
    model_name: str
    reference: str
    description: str
    input_name: str
    input_shape: list[int]
    preprocessing_steps: list[dict[str, Any]]
    postprocessing_steps: list[dict[str, Any]]
    dataset_eval_path: str
    artifact_base_path: str
    profile_choice: Literal["onnx only", "onnx + q-lite"]


DEFAULT_INPUT_SHAPE = [1, 3, 224, 224]
_UNSET = object()
BACK = object()


def normalize_family(family: str | None) -> str | None:
    """Normalize family value, converting empty strings to None."""
    if family is None:
        return None

    normalized = family.strip()
    return normalized if normalized else None


def validate_custom_task(task: str) -> bool:
    """Validate custom task input against path-safe identifier rules.

    Args:
        task: Custom task name from user input

    Returns:
        True if valid (matches IDENTIFIER_PATTERN), False otherwise
    """
    task = task.strip()
    return bool(task) and IDENTIFIER_PATTERN.fullmatch(task) is not None


def validate_custom_dataset(dataset: str) -> bool:
    """Validate custom dataset input against path-safe identifier rules.

    Args:
        dataset: Custom dataset name from user input

    Returns:
        True if valid (matches IDENTIFIER_PATTERN), False otherwise
    """
    dataset = dataset.strip()
    return bool(dataset) and IDENTIFIER_PATTERN.fullmatch(dataset) is not None


def validate_input_shape(shape_str: str) -> bool:
    """Validate input shape string."""
    try:
        parse_input_shape(shape_str)
        return True
    except (ValueError, TypeError):
        return False


def validate_family_identifier(value: str) -> bool:
    """Validate optional family identifier when a value is provided."""
    try:
        validate_identifier(value, "Family")
        return True
    except ValueError:
        return False


def validate_model_identifier(value: str) -> bool:
    """Validate model identifier for path-safe scaffold generation."""
    try:
        validate_identifier(value, "Model name")
        return True
    except ValueError:
        return False


def validate_postprocessing_identifier(value: str) -> bool:
    try:
        validate_identifier(value, "Postprocessing type")
        return True
    except ValueError:
        return False


def _get_builtin_models_root() -> Path:
    """Get the builtin models directory relative to package."""
    # Navigate from this file to the models directory
    # dx_modelzoo/tui/create_wizard.py -> dx_modelzoo/models
    return Path(__file__).parent.parent / "models"


def get_supported_domains() -> list[str]:
    """Get list of supported domains from builtin models."""
    models_root = _get_builtin_models_root()
    domains = []

    if models_root.exists():
        for item in sorted(models_root.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                domains.append(item.name)

    return domains


def get_supported_tasks(domain: str) -> list[str]:
    """Get list of supported tasks for a given domain from builtin models."""
    models_root = _get_builtin_models_root()
    domain_path = models_root / domain
    tasks = []

    if domain_path.exists():
        for item in sorted(domain_path.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                tasks.append(item.name)

    return tasks


def get_supported_datasets() -> list[str]:
    """Get list of registered dataset names."""
    return sorted(DATASET_REGISTRY.list())


def get_supported_evaluators() -> list[str]:
    """Get list of registered evaluator names."""
    return sorted(EVALUATOR_REGISTRY.list())


def get_preprocessing_options(
    input_shape: list[int] | None = None,
    available_steps: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str, bool]]:
    """Get list of preprocessing step options.

    Returns:
        List of (step_type, description, default_selected) tuples
    """
    effective_shape = input_shape or DEFAULT_INPUT_SHAPE
    steps = available_steps or _merge_observed_and_registry_steps(
        [],
        PREPROCESSING_REGISTRY,
        input_shape=effective_shape,
    )
    return _build_step_options(steps, default_first=True)


def get_postprocessing_options(
    available_steps: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str, bool]]:
    """Get list of postprocessing step options.

    Returns:
        List of (step_type, description, default_selected) tuples
    """
    steps = available_steps or _merge_observed_and_registry_steps(
        [],
        POSTPROCESSING_REGISTRY,
        input_shape=DEFAULT_INPUT_SHAPE,
    )
    return _build_step_options(steps, default_first=True)


def format_step_as_function(step: dict[str, Any]) -> str:
    """Format a preprocessing or postprocessing step dict as a function-like string.

    Args:
        step: Step dictionary with 'type' and optional parameters

    Returns:
        Function-like string representation, e.g., 'resize(size=256, mode="torchvision")'
    """
    step_type = step.get("type", "unknown")

    # Get all parameters except 'type'
    params = []
    for key, value in step.items():
        if key == "type":
            continue

        # Format value appropriately
        if isinstance(value, str):
            params.append(f'{key}="{value}"')
        elif isinstance(value, list):
            params.append(f"{key}={value}")
        else:
            params.append(f"{key}={value}")

    if params:
        return f"{step_type}({', '.join(params)})"
    else:
        return f"{step_type}()"


def _iter_builtin_model_dicts(domain: str, task: str) -> list[dict[str, Any]]:
    """Load builtin model YAML payloads for a domain/task, skipping invalid files."""
    models_root = _get_builtin_models_root()
    task_path = models_root / domain / task
    model_dicts = []

    if task_path.exists() and task_path.is_dir():
        for yaml_file in sorted(task_path.rglob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception:
                continue
            if isinstance(data, dict):
                model_dicts.append(data)

    return model_dicts


def _collect_task_steps(domain: str, task: str, section_name: str) -> list[dict[str, Any]]:
    """Collect unique step dicts from builtin YAMLs while preserving encounter order."""
    steps_dict: dict[str, dict[str, Any]] = {}

    for data in _iter_builtin_model_dicts(domain, task):
        section_steps = data.get(section_name, [])
        if not isinstance(section_steps, list):
            continue
        for step in section_steps:
            if not isinstance(step, dict) or "type" not in step:
                continue
            step_key = yaml.safe_dump(step, sort_keys=True)
            if step_key not in steps_dict:
                steps_dict[step_key] = deepcopy(step)

    return list(steps_dict.values())


def _ordered_registry_types(registry_names: list[str]) -> list[str]:
    """Preserve registry insertion order for non-task-specific options."""
    return list(registry_names)


def _synthesize_step_parameter(
    step_type: str,
    param_name: str,
    default: Any,
    input_shape: list[int],
) -> Any:
    """Build a usable parameter value from constructor defaults and lightweight heuristics."""
    if default is not inspect._empty:
        if default is not None:
            return deepcopy(default)
        if param_name in {"target_size"}:
            return [input_shape[-2], input_shape[-1]]
        return _UNSET

    if param_name == "height":
        return input_shape[-2]
    if param_name == "width":
        return input_shape[-1]
    if param_name == "size":
        return int(round(max(input_shape[-2], input_shape[-1]) / 0.875))
    if param_name == "x":
        return 255
    if param_name == "mean":
        return [0.485, 0.456, 0.406]
    if param_name == "std":
        return [0.229, 0.224, 0.225]
    if param_name == "axis":
        if step_type == "transpose":
            return [2, 0, 1]
        if step_type == "expanddim":
            return 0
    if param_name == "form":
        return "BGR2RGB"
    if param_name == "target_size":
        return [input_shape[-2], input_shape[-1]]
    if param_name == "input_size":
        return max(input_shape[-2], input_shape[-1])

    return _UNSET


def _apply_dynamic_step_defaults(step: dict[str, Any], step_type: str, input_shape: list[int]) -> None:
    """Add heuristic defaults for registry steps that hide parameters behind kwargs or None defaults."""
    if step_type == "resize":
        step.setdefault("mode", "torchvision")
        step.setdefault("size", int(round(max(input_shape[-2], input_shape[-1]) / 0.875)))
        step.setdefault("interpolation", "BILINEAR")
    elif step_type == "topk":
        step.setdefault("k", [1, 5])
    elif step_type == "nms":
        step.setdefault("conf_thres", 0.001)
        step.setdefault("iou_thres", 0.7)
        step.setdefault("max_output_boxes", 300)
    elif step_type == "segmentation_argmax":
        step.setdefault("layout", "nchw")
        step.setdefault("target_size", [input_shape[-2], input_shape[-1]])


def _synthesize_registered_step(
    step_type: str,
    registry,
    input_shape: list[int],
) -> dict[str, Any] | None:
    """Synthesize a usable config for a registry type from constructor signatures and heuristics."""
    cls = registry.get(step_type)
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        step = {"type": step_type}
        _apply_dynamic_step_defaults(step, step_type, input_shape)
        return step

    step = {"type": step_type}
    unresolved_required_params = set()

    for param_name, param in signature.parameters.items():
        if param_name == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        value = _synthesize_step_parameter(step_type, param_name, param.default, input_shape)
        if value is not _UNSET:
            step[param_name] = value
        elif param.default is inspect._empty:
            unresolved_required_params.add(param_name)

    _apply_dynamic_step_defaults(step, step_type, input_shape)

    if unresolved_required_params:
        return None

    return step


def _merge_observed_and_registry_steps(
    observed_steps: list[dict[str, Any]],
    registry,
    input_shape: list[int],
) -> list[dict[str, Any]]:
    """Return observed steps first, then synthesized registry-backed extras."""
    merged_steps = [deepcopy(step) for step in observed_steps]
    seen_types = {step.get("type") for step in merged_steps if step.get("type")}

    for step_type in _ordered_registry_types(registry.list()):
        if step_type in seen_types:
            continue
        synthesized = _synthesize_registered_step(step_type, registry, input_shape)
        if synthesized is None:
            continue
        merged_steps.append(synthesized)

    return merged_steps


def _build_step_options(
    available_steps: list[dict[str, Any]],
    default_selected_types: set[str] | None = None,
    default_first: bool = False,
) -> list[tuple[str, str, bool]]:
    """Convert step dicts to wizard multi-select options."""
    options = []
    seen_types = set()

    for step_dict in available_steps:
        step_type = step_dict.get("type")
        if not step_type or step_type in seen_types:
            continue
        is_default = (
            step_type in default_selected_types if default_selected_types is not None else default_first and not options
        )
        options.append((step_type, format_step_as_function(step_dict), is_default))
        seen_types.add(step_type)

    return options


def get_datasets_for_task(domain: str, task: str) -> list[str]:
    """Get dataset types used by models for a specific domain/task.

    Scans builtin YAML files under models/<domain>/<task> to find dataset types
    actually used by models. Falls back to all registered datasets if no task-specific
    datasets are found.

    Args:
        domain: Domain name (e.g., 'cv')
        task: Task name (e.g., 'image_classification')

    Returns:
        List of dataset type names sorted and deduplicated
    """
    datasets = set()

    for data in _iter_builtin_model_dicts(domain, task):
        dataset_info = data.get("dataset", {})
        if isinstance(dataset_info, dict):
            dataset_type = dataset_info.get("type")
            if dataset_type:
                datasets.add(dataset_type)

    # If we found task-specific datasets, return them
    if datasets:
        return sorted(datasets)

    # Otherwise fall back to all registered datasets
    return get_supported_datasets()


def get_preprocessing_for_task(domain: str, task: str, input_shape: list[int]) -> list[dict[str, Any]]:
    """Get preprocessing step dicts for a task, preferring observed YAML steps."""
    observed_steps = _collect_task_steps(domain, task, "preprocessing")
    return _merge_observed_and_registry_steps(
        observed_steps,
        PREPROCESSING_REGISTRY,
        input_shape=input_shape,
    )


def get_postprocessing_for_task(domain: str, task: str) -> list[dict[str, Any]]:
    """Get postprocessing step dicts for a task, preferring observed YAML steps.

    Scans builtin YAML files under models/<domain>/<task> to collect actual
    postprocessing steps used. Returns unique step dicts preserving real parameter values,
    then appends registry-backed synthesized defaults for additional available step types.

    Args:
        domain: Domain name (e.g., 'cv')
        task: Task name (e.g., 'image_classification')

    Returns:
        List of postprocessing step dictionaries
    """
    observed_steps = _collect_task_steps(domain, task, "postprocessing")
    if observed_steps:
        return observed_steps
    return _merge_observed_and_registry_steps(
        [],
        POSTPROCESSING_REGISTRY,
        input_shape=DEFAULT_INPUT_SHAPE,
    )


def build_preprocessing_steps(
    selected_types: list[str],
    input_shape: list[int],
    available_steps: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build preprocessing steps from selected step types.

    Args:
        selected_types: List of selected preprocessing step types
        input_shape: Input shape for shape-dependent steps (must have at least 2 dimensions)
        available_steps: Optional task-aware available step dicts to prefer.

    Returns:
        List of preprocessing step dictionaries

    Raises:
        ValueError: If input_shape has fewer than 2 dimensions
    """
    # Guard against invalid short input_shape values
    if len(input_shape) < 2:
        raise ValueError(f"input_shape must have at least 2 dimensions, got {len(input_shape)}")

    steps = []
    selected_set = set(selected_types)
    built_types = set()

    if available_steps:
        for step in available_steps:
            step_type = step.get("type")
            if step_type in selected_set and step_type not in built_types:
                steps.append(deepcopy(step))
                built_types.add(step_type)

    for step_type in selected_types:
        if step_type in built_types or step_type not in PREPROCESSING_REGISTRY:
            continue
        synthesized = _synthesize_registered_step(step_type, PREPROCESSING_REGISTRY, input_shape)
        if synthesized is None:
            continue
        steps.append(synthesized)
        built_types.add(step_type)

    return steps


def build_postprocessing_steps(
    selected_types: list[str], available_steps: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Build postprocessing steps from selected step types.

    Args:
        selected_types: List of selected postprocessing step types
        available_steps: Optional list of available step dicts to choose from.
                        If provided, will use these instead of templates.

    Returns:
        List of postprocessing step dictionaries
    """
    steps = []
    built_types = set()

    if available_steps:
        for step in available_steps:
            step_type = step.get("type")
            if step_type in selected_types and step_type not in built_types:
                steps.append(deepcopy(step))
                built_types.add(step_type)

    for step_type in selected_types:
        if step_type in built_types or step_type not in POSTPROCESSING_REGISTRY:
            continue
        synthesized = _synthesize_registered_step(step_type, POSTPROCESSING_REGISTRY, DEFAULT_INPUT_SHAPE)
        if synthesized is None:
            continue
        steps.append(synthesized)
        built_types.add(step_type)

    return steps


def result_to_dict(result: CreateWizardResult) -> dict[str, Any]:
    """Convert CreateWizardResult to a dictionary.

    Args:
        result: The wizard result object

    Returns:
        Dictionary representation of the result
    """
    return {
        "domain": result.domain,
        "task": result.task,
        "dataset_name": result.dataset_name,
        "family": normalize_family(result.family),  # Ensure family is normalized to None if blank
        "model_name": result.model_name,
        "reference": result.reference,
        "description": result.description,
        "input_name": result.input_name,
        "input_shape": result.input_shape,
        "preprocessing_steps": result.preprocessing_steps,
        "postprocessing_steps": result.postprocessing_steps,
        "dataset_eval_path": result.dataset_eval_path,
        "artifact_base_path": result.artifact_base_path,
        "profile_choice": result.profile_choice,
    }


# Prompt helpers for lightweight terminal interaction

console = Console()

ANSI_RESET = "\x1b[0m"
ANSI_BOLD_CYAN = "\x1b[1;36m"
ANSI_DIM = "\x1b[2m"
ANSI_GREEN = "\x1b[32m"
ANSI_RED = "\x1b[31m"
ANSI_BOLD = "\x1b[1m"


def _ansi(text: str, *codes: str) -> str:
    if not codes:
        return text
    return f"\x1b[{';'.join(codes)}m{text}{ANSI_RESET}"


def _write_terminal(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def format_step_header(step_index: int | None, total_steps: int | None, prompt: str) -> str:
    """Format a step header like '[1/12] Select domain:'."""
    if step_index is None or total_steps is None:
        return prompt
    return f"[{step_index}/{total_steps}] {prompt}"


def format_completion_summary(
    step_index: int | None,
    total_steps: int | None,
    label: str,
    value: str,
) -> str:
    """Format a compact completion summary line."""
    prefix = f"[{step_index}/{total_steps}] " if step_index is not None and total_steps is not None else ""
    return f"{prefix}{label}: {value}"


def _truncate_inline(text: str, limit: int = 120) -> str:
    """Keep compact summaries on a single line."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _interactive_picker_supported() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_key() -> str:
    """Read a single keypress from the terminal."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first == "\x03":
            raise KeyboardInterrupt
        if first in ("\r", "\n"):
            return "enter"
        if first in ("\x7f", "\b"):
            return "backspace"
        if first == " ":
            return "space"
        if first == "\x1b":
            second = sys.stdin.read(1)
            if second != "[":
                return "ignore"
            third = sys.stdin.read(1)
            if third == "A":
                return "up"
            if third == "B":
                return "down"
            return "ignore"
        if first.isprintable():
            return first
        return "ignore"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _save_cursor_position() -> None:
    _write_terminal("\x1b7")


def _restore_cursor_and_clear() -> None:
    _write_terminal("\x1b8\x1b[J")


def _build_picker_option_lines(choices: list[str], current_index: int) -> list[str]:
    lines = []
    for idx, choice in enumerate(choices):
        if idx == current_index:
            lines.append(f" {_ansi('❯', '32')} {_ansi(choice, '1')}")
        else:
            lines.append(_ansi(f"   {choice}", "2"))
    return lines


def _build_multiselect_option_lines(
    options: list[tuple[str, str, bool]],
    current_index: int,
    selected: set[str],
) -> list[str]:
    lines = []
    for idx, (step_type, description, _default) in enumerate(options):
        pointer = _ansi("❯", "32") if idx == current_index else _ansi(" ", "2")
        marker = _ansi("✓", "32") if step_type in selected else _ansi("·", "2")
        description_text = _ansi(description, "1") if idx == current_index else _ansi(description, "2")
        lines.append(f" {pointer} {marker} {description_text}")
    return lines


def _render_text_prompt_block(header: str, default: str, error_message: str | None = None) -> None:
    default_text = f" {_ansi(f'(default: {default})', '2')}" if default else ""
    lines = [f"{_ansi(header, '1', '36')}{default_text}"]
    if error_message:
        lines.append(_ansi(error_message, "31"))
    _write_terminal("\n".join(lines) + "\n")


def _render_text_entry_block(
    header: str,
    value: str,
    default: str,
    current_index: int,
    error_message: str | None = None,
) -> None:
    default_text = f" {_ansi(f'(default: {default})', '2')}" if default else ""
    lines = [
        f"{_ansi(header, '1', '36')}{default_text}",
        _ansi("Type to edit, ↑↓ to move, Enter to confirm, Backspace to delete, Ctrl+C to cancel", "2"),
    ]
    if error_message:
        lines.append(_ansi(error_message, "31"))
    input_value = value if value else _ansi("(empty)", "2")
    if current_index == 0:
        lines.append(f" {_ansi('❯', '32')} {_ansi('Input', '1')}: {input_value}")
        lines.append(_ansi("   [Back]", "2"))
    else:
        lines.append(f"{_ansi('   Input', '2')}: {_ansi(value if value else '(empty)', '2')}")
        lines.append(f" {_ansi('❯', '32')} {_ansi('[Back]', '1')}")
    _write_terminal("\n".join(lines) + "\n")


def _format_text_summary_value(value: str, *, default: str, required: bool) -> str:
    if not value.strip() and not required and not default:
        return "(skipped)"
    if default and value == default:
        return f"(default: {default})"
    return _truncate_inline(value)


ANSI_PATTERN = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|[@-_])")


def _strip_ansi(text: str) -> str:
    return ANSI_PATTERN.sub("", text)


def _count_terminal_lines(lines: list[str]) -> int:
    width = max(shutil.get_terminal_size(fallback=(80, 24)).columns, 1)
    total = 0
    for line in lines:
        plain = _strip_ansi(line)
        total += max(1, math.ceil(len(plain) / width))
    return total


def _clear_rendered_lines(line_count: int) -> None:
    for _ in range(line_count):
        _write_terminal("\x1b[1A\x1b[2K\r")


def _clear_last_completion_summary() -> None:
    _clear_rendered_lines(1)


def _render_picker_block(
    header: str,
    option_lines: list[str],
    instructions: str,
) -> None:
    lines = [_ansi(header, "1", "36")]
    if instructions:
        lines.append(_ansi(instructions, "2"))
    lines.extend(option_lines)
    _write_terminal("\n".join(lines) + "\n")


def _interactive_select(
    header: str,
    choices: list[str],
    *,
    default: str | None = None,
) -> str | None:
    current_index = choices.index(default) if default in choices else 0
    instructions = "Use ↑↓ to move, Enter to select, Ctrl+C to cancel"
    rendered_line_count = 0
    _write_terminal("\n")
    needs_render = True
    try:
        while True:
            if needs_render:
                option_lines = _build_picker_option_lines(choices, current_index)
                block_lines = [_ansi(header, "1", "36"), _ansi(instructions, "2"), *option_lines]
                if rendered_line_count:
                    _clear_rendered_lines(rendered_line_count)
                _render_picker_block(header, option_lines, instructions)
                rendered_line_count = _count_terminal_lines(block_lines)
                needs_render = False

            key = _read_key()
            if key == "up":
                current_index = (current_index - 1) % len(choices)
                needs_render = True
            elif key == "down":
                current_index = (current_index + 1) % len(choices)
                needs_render = True
            elif key == "enter":
                if rendered_line_count:
                    _clear_rendered_lines(rendered_line_count + 1)
                return choices[current_index]
    except (KeyboardInterrupt, EOFError):
        if rendered_line_count:
            _clear_rendered_lines(rendered_line_count + 1)
        return None


def _interactive_multiselect(
    header: str,
    options: list[tuple[str, str, bool]],
    *,
    initial_selected: list[str] | None = None,
    enable_back: bool = False,
) -> list[str] | object | None:
    current_index = 0
    selected = (
        set(initial_selected)
        if initial_selected is not None
        else {step_type for step_type, _description, is_default in options if is_default}
    )
    instructions = "Use ↑↓ to move, Space to toggle, Enter to confirm, Ctrl+C to cancel"
    rendered_line_count = 0
    _write_terminal("\n")
    needs_render = True
    try:
        while True:
            if needs_render:
                option_lines = _build_multiselect_option_lines(options, current_index, selected)
                block_lines = [_ansi(header, "1", "36"), _ansi(instructions, "2"), *option_lines]
                if rendered_line_count:
                    _clear_rendered_lines(rendered_line_count)
                _render_picker_block(header, option_lines, instructions)
                rendered_line_count = _count_terminal_lines(block_lines)
                needs_render = False

            key = _read_key()
            if key == "up":
                current_index = (current_index - 1) % len(options)
                needs_render = True
            elif key == "down":
                current_index = (current_index + 1) % len(options)
                needs_render = True
            elif key == "space":
                step_type = options[current_index][0]
                if enable_back and step_type == "[Back]":
                    needs_render = True
                    continue
                if step_type in selected:
                    selected.remove(step_type)
                else:
                    selected.add(step_type)
                needs_render = True
            elif key == "enter":
                if rendered_line_count:
                    _clear_rendered_lines(rendered_line_count + 1)
                if enable_back and options[current_index][0] == "[Back]":
                    return BACK
                return [step_type for step_type, _description, _default in options if step_type in selected]
    except (KeyboardInterrupt, EOFError):
        if rendered_line_count:
            _clear_rendered_lines(rendered_line_count + 1)
        return None


def prompt_select(
    prompt: str,
    choices: list[str],
    default: str | None = None,
    *,
    step_index: int | None = None,
    total_steps: int | None = None,
    summary_label: str | None = None,
    enable_back: bool = False,
) -> str | object | None:
    """Present numbered choices and return selection.

    Args:
        prompt: The question to ask
        choices: List of available choices
        default: Default choice (optional)

    Returns:
        Selected choice or None if cancelled (Ctrl+C or empty with no default)
    """
    header = format_step_header(step_index, total_steps, prompt)
    prompt_choices = [*choices, "[Back]"] if enable_back else choices
    if _interactive_picker_supported():
        selected = _interactive_select(header, prompt_choices, default=default)
        if selected == "[Back]":
            return BACK
        if selected is not None and summary_label:
            _write_terminal(
                f"✓ {format_completion_summary(step_index, total_steps, summary_label, _truncate_inline(selected))}\n"
            )
        return selected

    console.print(f"\n[bold cyan]{header}[/bold cyan]")
    for idx, choice in enumerate(prompt_choices, 1):
        console.print(f"  {idx}. {choice}")

    default_text = f" (default: {default})" if default else ""
    prompt_text = f"Select 1-{len(prompt_choices)}{default_text}"

    try:
        while True:
            selection = Prompt.ask(prompt_text, default=default or "")

            # Handle empty input
            if not selection.strip():
                if default:
                    # Find default in choices
                    if default in choices:
                        if summary_label:
                            completion = format_completion_summary(
                                step_index,
                                total_steps,
                                summary_label,
                                _truncate_inline(default),
                            )
                            console.print(f"[green]✓[/green] {completion}")
                        return default
                    return None
                else:
                    # No default, cancellation
                    return None

            # Try numeric selection
            try:
                idx = int(selection)
                if 1 <= idx <= len(prompt_choices):
                    if enable_back and prompt_choices[idx - 1] == "[Back]":
                        return BACK
                    if summary_label:
                        completion = format_completion_summary(
                            step_index,
                            total_steps,
                            summary_label,
                            _truncate_inline(prompt_choices[idx - 1]),
                        )
                        console.print(f"[green]✓[/green] {completion}")
                    return prompt_choices[idx - 1]
                else:
                    console.print(f"[red]Please enter a number between 1 and {len(prompt_choices)}[/red]")
            except ValueError:
                # Try text match
                selection_lower = selection.lower()
                matches = [c for c in prompt_choices if c.lower() == selection_lower]
                if matches:
                    if enable_back and matches[0] == "[Back]":
                        return BACK
                    if summary_label:
                        completion = format_completion_summary(
                            step_index,
                            total_steps,
                            summary_label,
                            _truncate_inline(matches[0]),
                        )
                        console.print(f"[green]✓[/green] {completion}")
                    return matches[0]

                # Fuzzy match
                matches = [c for c in prompt_choices if selection_lower in c.lower()]
                if len(matches) == 1:
                    if enable_back and matches[0] == "[Back]":
                        return BACK
                    if summary_label:
                        completion = format_completion_summary(
                            step_index,
                            total_steps,
                            summary_label,
                            _truncate_inline(matches[0]),
                        )
                        console.print(f"[green]✓[/green] {completion}")
                    return matches[0]
                elif len(matches) > 1:
                    console.print(f"[yellow]Ambiguous: {', '.join(matches)}[/yellow]")
                else:
                    console.print(f"[red]Invalid choice: {selection}[/red]")

    except (KeyboardInterrupt, EOFError):
        return None


def prompt_text(
    prompt: str,
    default: str = "",
    required: bool = True,
    validator: callable = None,
    validator_message: str | None = None,
    *,
    step_index: int | None = None,
    total_steps: int | None = None,
    summary_label: str | None = None,
    enable_back: bool = False,
) -> str | object | None:
    """Prompt for text input with optional validation.

    Args:
        prompt: The question to ask
        default: Default value
        required: Whether input is required
        validator: Optional validation function that returns bool
        validator_message: Optional custom validation error message

    Returns:
        User input or None if cancelled
    """
    default_text = f" [dim](default: {default})[/dim]" if default else ""
    full_prompt = f"[bold cyan]{format_step_header(step_index, total_steps, prompt)}[/bold cyan]{default_text}"

    try:
        if _interactive_picker_supported() and enable_back:
            header = format_step_header(step_index, total_steps, prompt)
            error_message = None
            current_index = 0
            value = default
            _save_cursor_position()
            while True:
                _restore_cursor_and_clear()
                _write_terminal("\n")
                _render_text_entry_block(header, value, default, current_index, error_message)

                block_lines = [
                    f"{header}{f' (default: {default})' if default else ''}",
                    "Type to edit, ↑↓ to move, Enter to confirm, Backspace to delete, Ctrl+C to cancel",
                    *([error_message] if error_message else []),
                    f"입력하기: {value if value else '(empty)'}",
                    "[Back]",
                ]
                rendered_line_count = _count_terminal_lines(block_lines)
                key = _read_key()

                if key == "up":
                    current_index = (current_index - 1) % 2
                    error_message = None
                    continue
                if key == "down":
                    current_index = (current_index + 1) % 2
                    error_message = None
                    continue
                if current_index == 0 and key == "backspace":
                    value = value[:-1]
                    error_message = None
                    continue
                if key == "enter":
                    if current_index == 1:
                        _clear_rendered_lines(rendered_line_count + 1)
                        return BACK
                    if not value.strip():
                        if not required and not default:
                            if summary_label:
                                _clear_rendered_lines(rendered_line_count + 1)
                                summary_value = _format_text_summary_value("", default=default, required=required)
                                completion = format_completion_summary(
                                    step_index,
                                    total_steps,
                                    summary_label,
                                    summary_value,
                                )
                                _write_terminal(f"✓ {completion}\n")
                            return ""
                        if default:
                            if summary_label:
                                _clear_rendered_lines(rendered_line_count + 1)
                                summary_value = _format_text_summary_value(default, default=default, required=required)
                                completion = format_completion_summary(
                                    step_index,
                                    total_steps,
                                    summary_label,
                                    summary_value,
                                )
                                _write_terminal(f"✓ {completion}\n")
                            return default
                        error_message = "This field is required"
                        continue
                    if validator and not validator(value):
                        error_message = validator_message or "Invalid input, please try again"
                        continue
                    if summary_label:
                        _clear_rendered_lines(rendered_line_count + 1)
                        summary_value = _format_text_summary_value(value, default=default, required=required)
                        completion = format_completion_summary(
                            step_index,
                            total_steps,
                            summary_label,
                            summary_value,
                        )
                        _write_terminal(f"✓ {completion}\n")
                    return value
                if current_index == 0 and len(key) == 1 and key not in {"\r", "\n"}:
                    value += key
                    error_message = None
                    continue
                error_message = None

        if _interactive_picker_supported():
            header = format_step_header(step_index, total_steps, prompt)
            error_message = None
            _save_cursor_position()
            while True:
                _restore_cursor_and_clear()
                _write_terminal("\n")
                _render_text_prompt_block(header, default, error_message)
                value = Prompt.ask("", default=default)

                if not value.strip():
                    if not required and not default:
                        if summary_label:
                            _restore_cursor_and_clear()
                            summary_value = _format_text_summary_value("", default=default, required=required)
                            completion = format_completion_summary(
                                step_index,
                                total_steps,
                                summary_label,
                                summary_value,
                            )
                            _write_terminal(f"✓ {completion}\n")
                        return ""
                    if default:
                        if summary_label:
                            _restore_cursor_and_clear()
                            summary_value = _format_text_summary_value(default, default=default, required=required)
                            completion = format_completion_summary(
                                step_index,
                                total_steps,
                                summary_label,
                                summary_value,
                            )
                            _write_terminal(f"✓ {completion}\n")
                        return default
                    error_message = "This field is required"
                    continue

                if validator and not validator(value):
                    error_message = validator_message or "Invalid input, please try again"
                    continue

                if summary_label:
                    _restore_cursor_and_clear()
                    summary_value = _format_text_summary_value(value, default=default, required=required)
                    completion = format_completion_summary(
                        step_index,
                        total_steps,
                        summary_label,
                        summary_value,
                    )
                    _write_terminal(f"✓ {completion}\n")
                return value

        while True:
            console.print(f"\n{full_prompt}")
            value = Prompt.ask("", default=default)

            # Handle empty input
            if not value.strip():
                if not required and not default:
                    if summary_label:
                        completion = format_completion_summary(
                            step_index,
                            total_steps,
                            summary_label,
                            _format_text_summary_value("", default=default, required=required),
                        )
                        console.print(f"[green]✓[/green] {completion}")
                    return ""
                elif default:
                    if summary_label:
                        completion = format_completion_summary(
                            step_index,
                            total_steps,
                            summary_label,
                            _format_text_summary_value(default, default=default, required=required),
                        )
                        console.print(f"[green]✓[/green] {completion}")
                    return default
                else:
                    console.print("[red]This field is required[/red]")
                    continue

            # Validate if validator provided
            if validator:
                if validator(value):
                    if summary_label:
                        completion = format_completion_summary(
                            step_index,
                            total_steps,
                            summary_label,
                            _format_text_summary_value(value, default=default, required=required),
                        )
                        console.print(f"[green]✓[/green] {completion}")
                    return value
                else:
                    console.print(f"[red]{validator_message or 'Invalid input, please try again'}[/red]")
                    continue

            if summary_label:
                completion = format_completion_summary(
                    step_index,
                    total_steps,
                    summary_label,
                    _format_text_summary_value(value, default=default, required=required),
                )
                console.print(f"[green]✓[/green] {completion}")
            return value

    except (KeyboardInterrupt, EOFError):
        return None


def prompt_multiselect(
    prompt: str,
    options: list[tuple[str, str, bool]],
    *,
    step_index: int | None = None,
    total_steps: int | None = None,
    summary_label: str | None = None,
    selected: list[str] | None = None,
    enable_back: bool = False,
) -> list[str] | object | None:
    """Present multi-select choices.

    Args:
        prompt: The question to ask
        options: List of (step_type, description, default_selected) tuples

    Returns:
        List of selected step types or None if cancelled
    """
    header = format_step_header(step_index, total_steps, prompt)
    prompt_options = [*options, ("[Back]", "[Back]", False)] if enable_back else options

    def _print_multiselect_summary(selected_types: list[str]) -> None:
        if not summary_label:
            return
        selected_labels = [description for step_type, description, _default in options if step_type in selected_types]
        summary_value = _truncate_inline(", ".join(selected_labels) if selected_labels else "(none)")
        console.print(
            f"[green]✓[/green] {format_completion_summary(step_index, total_steps, summary_label, summary_value)}"
        )

    if _interactive_picker_supported():
        selected_result = _interactive_multiselect(
            header,
            prompt_options,
            initial_selected=selected,
            enable_back=enable_back,
        )
        if selected_result is BACK:
            return BACK
        if selected_result is not None and summary_label:
            selected_labels = [
                description for step_type, description, _default in options if step_type in selected_result
            ]
            summary_value = _truncate_inline(", ".join(selected_labels) if selected_labels else "(none)")
            _write_terminal(f"✓ {format_completion_summary(step_index, total_steps, summary_label, summary_value)}\n")
        return selected_result

    console.print(f"\n[bold cyan]{header}[/bold cyan]")
    console.print("[dim]Enter numbers separated by commas, or 'all', or 'none'[/dim]")

    # Show options with defaults marked
    default_indices = []
    effective_selected = set(selected) if selected is not None else None
    for idx, (step_type, description, is_default) in enumerate(prompt_options, 1):
        is_selected = step_type in effective_selected if effective_selected is not None else is_default
        marker = "[green]✓[/green]" if is_selected else " "
        console.print(f"  {marker} {idx}. {description}")
        if is_selected:
            default_indices.append(idx)

    default_text = ",".join(map(str, default_indices)) if default_indices else "none"

    try:
        while True:
            selection = Prompt.ask(f"Select options (default: {default_text})", default=default_text)
            if enable_back and selection.lower() == "back":
                return BACK

            if selection.lower() == "none":
                _print_multiselect_summary([])
                return []

            if selection.lower() == "all":
                result = [opt[0] for opt in options]
                _print_multiselect_summary(result)
                return result

            # Parse comma-separated indices
            try:
                if not selection.strip():
                    # Use defaults
                    indices = default_indices
                else:
                    parts = [p.strip() for p in selection.split(",")]
                    indices = [int(p) for p in parts if p]

                # Validate all indices
                if all(1 <= idx <= len(prompt_options) for idx in indices):
                    if enable_back and len(indices) == 1 and prompt_options[indices[0] - 1][0] == "[Back]":
                        return BACK
                    result = [prompt_options[idx - 1][0] for idx in indices if prompt_options[idx - 1][0] != "[Back]"]
                    _print_multiselect_summary(result)
                    return result
                else:
                    console.print(f"[red]Please enter numbers between 1 and {len(prompt_options)}[/red]")
            except ValueError:
                console.print("[red]Invalid input. Use comma-separated numbers (e.g., 1,2,3)[/red]")

    except (KeyboardInterrupt, EOFError):
        return None


def run_create_wizard() -> CreateWizardResult | None:
    """Run the create wizard and return the result.

    Uses sequential lightweight prompts to collect all necessary information
    for creating a custom model scaffold.

    Returns:
        CreateWizardResult if wizard completed successfully, None if cancelled
    """
    console.print("\n[bold green]dx-modelzoo Create Wizard[/bold green]")
    console.print("[dim]Press Ctrl+C at any time to cancel[/dim]\n")
    total_steps = 12

    domains = get_supported_domains()
    if not domains:
        console.print("[red]Error: No domains found in builtin models[/red]")
        return None

    domain = domains[0] if len(domains) == 1 else None
    task = None
    task_choice = None
    dataset_name = None
    dataset_choice = None
    family_input = ""
    model_name = ""
    input_name = "input"
    input_shape_str = "1,3,224,224"
    preprocessing_selected = None
    postprocessing_selected = None
    custom_postprocessing_type = None
    dataset_eval_path = "/path/to/dataset/val"
    artifact_base_path = "/path/to/artifacts"
    profile_choice_result = "onnx only"

    step = 1
    while True:
        if step == 1:
            result = prompt_select(
                "Select domain:",
                domains,
                default=domain,
                step_index=1,
                total_steps=total_steps,
                summary_label="Domain",
            )
            if result is None:
                return None
            domain = result
            step = 2
            continue

        if step == 2:
            tasks = get_supported_tasks(domain)
            evaluators = get_supported_evaluators()
            all_tasks = sorted(set(tasks + evaluators))
            all_tasks.append("[Custom]")
            result = prompt_select(
                "Select task:",
                all_tasks,
                default=task_choice
                if task_choice in all_tasks
                else ("[Custom]" if task_choice == "[Custom]" else None),
                step_index=2,
                total_steps=total_steps,
                summary_label="Task",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 1
                continue
            task_choice = result
            custom_postprocessing_type = None
            if task_choice == "[Custom]":
                custom_result = prompt_text(
                    "Enter custom task name:",
                    default=task or "",
                    required=True,
                    validator=validate_custom_task,
                    step_index=2,
                    total_steps=total_steps,
                    summary_label="Task",
                    enable_back=True,
                )
                if custom_result is None:
                    return None
                if custom_result is BACK:
                    _clear_last_completion_summary()
                    continue
                task = custom_result
            else:
                task = task_choice
            step = 3
            continue

        if step == 3:
            task_datasets = get_datasets_for_task(domain, task)
            task_datasets.append("[Custom]")
            result = prompt_select(
                "Select dataset:",
                task_datasets,
                default=dataset_choice
                if dataset_choice in task_datasets
                else ("[Custom]" if dataset_choice == "[Custom]" else None),
                step_index=3,
                total_steps=total_steps,
                summary_label="Dataset",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 2
                continue
            dataset_choice = result
            if dataset_choice == "[Custom]":
                custom_result = prompt_text(
                    "Enter custom dataset name:",
                    default=dataset_name or "",
                    required=True,
                    validator=validate_custom_dataset,
                    step_index=3,
                    total_steps=total_steps,
                    summary_label="Dataset",
                    enable_back=True,
                )
                if custom_result is None:
                    return None
                if custom_result is BACK:
                    _clear_last_completion_summary()
                    continue
                dataset_name = custom_result
            else:
                dataset_name = dataset_choice
            step = 4
            continue

        if step == 4:
            result = prompt_text(
                "Model family (optional, leave empty to skip):",
                default=family_input,
                required=False,
                validator=validate_family_identifier,
                validator_message=(
                    "Family must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ " "and cannot contain spaces or path separators."
                ),
                step_index=4,
                total_steps=total_steps,
                summary_label="Family",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 3
                continue
            family_input = result
            step = 5
            continue

        if step == 5:
            result = prompt_text(
                "Model name:",
                default=model_name,
                required=True,
                validator=validate_model_identifier,
                validator_message=(
                    "Model name must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ "
                    "and cannot contain spaces or path separators."
                ),
                step_index=5,
                total_steps=total_steps,
                summary_label="Model name",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 4
                continue
            model_name = result
            step = 6
            continue

        if step == 6:
            result = prompt_text(
                "Input tensor name:",
                default=input_name,
                step_index=6,
                total_steps=total_steps,
                summary_label="Input tensor name",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 5
                continue
            input_name = result
            step = 7
            continue

        if step == 7:
            result = prompt_text(
                "Input shape (2+ integers, comma/space separated):",
                default=input_shape_str,
                validator=validate_input_shape,
                step_index=7,
                total_steps=total_steps,
                summary_label="Input shape",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 6
                continue
            input_shape_str = result
            step = 8
            continue

        input_shape = parse_input_shape(input_shape_str)

        if step == 8:
            task_pre_steps = get_preprocessing_for_task(domain, task, input_shape)
            preprocessing_options = get_preprocessing_options(
                input_shape=input_shape,
                available_steps=task_pre_steps,
            )
            result = prompt_multiselect(
                "Select preprocessing steps:",
                preprocessing_options,
                step_index=8,
                total_steps=total_steps,
                summary_label="Preprocessing",
                selected=preprocessing_selected,
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 7
                continue
            preprocessing_selected = result
            step = 9
            continue

        if step == 9:
            task_post_steps = get_postprocessing_for_task(domain, task)
            postprocessing_options = get_postprocessing_options(available_steps=task_post_steps)
            postprocessing_options = [*postprocessing_options, ("[Custom]", "[Custom]", False)]
            result = prompt_multiselect(
                "Select postprocessing steps:",
                postprocessing_options,
                step_index=9,
                total_steps=total_steps,
                summary_label="Postprocessing",
                selected=postprocessing_selected,
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 8
                continue
            postprocessing_selected = result
            custom_postprocessing_type = None
            if "[Custom]" in postprocessing_selected:
                postprocessing_selected = [
                    step_type for step_type in postprocessing_selected if step_type != "[Custom]"
                ]
                custom_result = prompt_text(
                    "Enter custom postprocessing type:",
                    default=custom_postprocessing_type or "",
                    required=True,
                    validator=validate_postprocessing_identifier,
                    validator_message=(
                        "Postprocessing type must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ "
                        "and cannot contain spaces or path separators."
                    ),
                    step_index=9,
                    total_steps=total_steps,
                    summary_label="Custom postprocessing",
                    enable_back=True,
                )
                if custom_result is None:
                    return None
                if custom_result is BACK:
                    _clear_last_completion_summary()
                    continue
                custom_postprocessing_type = custom_result
            step = 10
            continue

        if step == 10:
            result = prompt_text(
                "Dataset evaluation path:",
                default=dataset_eval_path,
                step_index=10,
                total_steps=total_steps,
                summary_label="Dataset evaluation path",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 9
                continue
            dataset_eval_path = result
            step = 11
            continue

        if step == 11:
            result = prompt_text(
                "Artifact base path:",
                default=artifact_base_path,
                step_index=11,
                total_steps=total_steps,
                summary_label="Artifact base path",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 10
                continue
            artifact_base_path = result
            step = 12
            continue

        if step == 12:
            result = prompt_select(
                "Profile choice:",
                ["onnx only", "onnx + q-lite"],
                default=profile_choice_result,
                step_index=12,
                total_steps=total_steps,
                summary_label="Profile",
                enable_back=True,
            )
            if result is None:
                return None
            if result is BACK:
                _clear_last_completion_summary()
                step = 11
                continue
            profile_choice_result = result
            break

    # Build steps from selections
    family = normalize_family(family_input)
    preprocessing_steps = build_preprocessing_steps(preprocessing_selected, input_shape, task_pre_steps)
    postprocessing_steps = build_postprocessing_steps(postprocessing_selected, task_post_steps)
    if custom_postprocessing_type is not None:
        postprocessing_steps.append({"type": custom_postprocessing_type})

    # Summary
    console.print("\n[bold green]Configuration Summary:[/bold green]")
    console.print(f"  Domain: {domain}")
    console.print(f"  Task: {task}")
    console.print(f"  Dataset: {dataset_name}")
    console.print(f"  Family: {family or '(none)'}")
    console.print(f"  Model: {model_name}")
    console.print(f"  Input: {input_name} {input_shape}")
    console.print("  Preprocessing steps:")
    for step in preprocessing_steps:
        console.print(f"    - {format_step_as_function(step)}")
    console.print("  Postprocessing steps:")
    for step in postprocessing_steps:
        console.print(f"    - {format_step_as_function(step)}")
    console.print(f"  Profile: {profile_choice_result}")

    try:
        confirm = Confirm.ask("\nCreate model scaffold with this configuration?", default=True)
        if not confirm:
            return None
    except (KeyboardInterrupt, EOFError):
        return None

    return CreateWizardResult(
        domain=domain,
        task=task,
        dataset_name=dataset_name,
        family=family,
        model_name=model_name,
        reference="",  # No longer prompted - empty string
        description="",  # No longer prompted - empty string
        input_name=input_name,
        input_shape=input_shape,
        preprocessing_steps=preprocessing_steps,
        postprocessing_steps=postprocessing_steps,
        dataset_eval_path=dataset_eval_path,
        artifact_base_path=artifact_base_path,
        profile_choice=profile_choice_result,
    )
