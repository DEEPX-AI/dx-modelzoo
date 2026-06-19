from __future__ import annotations

import re
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from dx_modelzoo.loader.discovery import scan_all_models

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PROFILE_CHOICES = ("onnx only", "onnx + q-lite")
PREPROCESSING_PRESETS = {
    "imagenet": lambda shape: [
        {
            "type": "resize",
            "mode": "torchvision",
            "size": _default_resize(shape),
            "interpolation": "BILINEAR",
        },
        {
            "type": "centercrop",
            "height": shape[-2],
            "width": shape[-1],
        },
        {
            "type": "convertcolor",
            "form": "BGR2RGB",
        },
        {
            "type": "div",
            "x": 255,
        },
        {
            "type": "normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        {
            "type": "transpose",
            "axis": [2, 0, 1],
        },
        {
            "type": "expanddim",
            "axis": 0,
        },
    ]
}
POSTPROCESSING_PRESETS = {
    "topk": [
        {
            "type": "topk",
            "k": [1, 5],
        }
    ]
}


class InvalidIdentifierError(ValueError):
    """Raised when family or model identifiers are not path-safe."""


class DuplicateCustomModelError(ValueError):
    """Raised when a custom model name already exists at another path."""


@dataclass(frozen=True, init=False)
class ClassificationScaffold:
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

    def __init__(
        self,
        *,
        domain: str = "cv",
        task: str = "classification",
        dataset_name: str = "ILSVRC2012",
        family: str | None = None,
        model_name: str,
        reference: str,
        description: str,
        input_name: str,
        input_shape: list[int],
        preprocessing_steps: list[dict[str, Any]] | None = None,
        postprocessing_steps: list[dict[str, Any]] | None = None,
        dataset_eval_path: str,
        artifact_base_path: str,
        profile_choice: Literal["onnx only", "onnx + q-lite"] = "onnx only",
        preprocessing_preset: str | None = None,
        postprocessing_preset: str | None = None,
    ) -> None:
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "input_name", input_name)
        object.__setattr__(self, "input_shape", list(input_shape))
        object.__setattr__(
            self,
            "preprocessing_steps",
            _resolve_preprocessing_steps(input_shape, preprocessing_steps, preprocessing_preset),
        )
        object.__setattr__(
            self,
            "postprocessing_steps",
            _resolve_postprocessing_steps(postprocessing_steps, postprocessing_preset),
        )
        object.__setattr__(self, "dataset_eval_path", dataset_eval_path)
        object.__setattr__(self, "artifact_base_path", artifact_base_path)
        object.__setattr__(self, "profile_choice", profile_choice)


def _default_resize(shape: list[int]) -> int:
    crop_size = max(shape[-2], shape[-1])
    return int(round(crop_size / 0.875))


def validate_identifier(value: str, field_name: str) -> str:
    value = value.strip()
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise InvalidIdentifierError(
            f"{field_name} must match {IDENTIFIER_PATTERN.pattern} and cannot contain spaces or path separators."
        )
    return value


def _validate_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _resolve_preprocessing_steps(
    input_shape: list[int],
    preprocessing_steps: list[dict[str, Any]] | None,
    preprocessing_preset: str | None,
) -> list[dict[str, Any]]:
    if preprocessing_steps is not None:
        return deepcopy(preprocessing_steps)

    preset_name = preprocessing_preset or "imagenet"
    preprocessing_factory = PREPROCESSING_PRESETS.get(preset_name)
    if preprocessing_factory is None:
        raise ValueError(f"Unsupported preprocessing preset: {preset_name}")
    return preprocessing_factory(input_shape)


def _resolve_postprocessing_steps(
    postprocessing_steps: list[dict[str, Any]] | None,
    postprocessing_preset: str | None,
) -> list[dict[str, Any]]:
    if postprocessing_steps is not None:
        return deepcopy(postprocessing_steps)

    preset_name = postprocessing_preset or "topk"
    resolved_steps = POSTPROCESSING_PRESETS.get(preset_name)
    if resolved_steps is None:
        raise ValueError(f"Unsupported postprocessing preset: {preset_name}")
    return deepcopy(resolved_steps)


def parse_input_shape(raw_value: str) -> list[int]:
    normalized = raw_value.strip()
    if not normalized:
        raise ValueError("input shape cannot be blank")

    if re.fullmatch(r"\d+(?:(?:\s*,\s*|\s+)\d+)*", normalized) is None:
        raise ValueError("expected a list of positive integers using only digits, commas, and spaces, like 1,3,224,224")

    parts = re.split(r"[\s,]+", normalized)
    if len(parts) < 2:
        raise ValueError("input shape must include at least two dimensions")

    shape = [int(part) for part in parts]
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("all input shape dimensions must be positive integers")

    return shape


def build_custom_model_path(
    custom_root: Path,
    family: str | None,
    model_name: str,
    domain: str = "cv",
    task: str = "classification",
) -> Path:
    safe_domain = validate_identifier(domain, "Domain")
    safe_task = validate_identifier(task, "Task")
    safe_model_name = validate_identifier(model_name, "Model name")
    family_folder = safe_model_name if family is None else validate_identifier(family, "Family")
    return Path(custom_root) / safe_domain / safe_task / family_folder / f"{safe_model_name}.yaml"


def _profiles_for(choice: str) -> dict[str, Any]:
    if choice not in PROFILE_CHOICES:
        raise ValueError(f"Unsupported profile choice: {choice}")

    profiles = {
        "onnx": {
            "target": "onnx",
            "runtime": {
                "device": "gpu",
                "batch_size": 1,
            },
        }
    }

    if choice == "onnx + q-lite":
        profiles["q-lite"] = {
            "target": "dxnn",
            "compile": {
                "quantization": {
                    "lite": {
                        "num_samples": 100,
                        "method": "ema",
                    },
                },
            },
            "runtime": {
                "device": 0,
                "async": True,
            },
        }

    return profiles


def render_classification_config(spec: ClassificationScaffold) -> dict[str, Any]:
    return {
        "name": validate_identifier(spec.model_name, "Model name"),
        "task": validate_identifier(spec.task, "Task"),
        "reference": spec.reference,
        "description": spec.description,
        "evaluator": {
            "type": validate_identifier(spec.task, "Task"),
        },
        "inputs": [
            {
                "name": _validate_non_empty(spec.input_name, "Input name"),
                "shape": spec.input_shape,
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "preprocessing": deepcopy(spec.preprocessing_steps),
        "postprocessing": deepcopy(spec.postprocessing_steps),
        "dataset": {
            "type": validate_identifier(spec.dataset_name, "Dataset name"),
            "eval_path": spec.dataset_eval_path,
        },
        "artifacts": {
            "path": spec.artifact_base_path,
        },
        "profiles": _profiles_for(spec.profile_choice),
    }


def render_classification_yaml(spec: ClassificationScaffold) -> str:
    return yaml.safe_dump(render_classification_config(spec), sort_keys=False, allow_unicode=True)


def _needs_custom_dataset_template(spec: ClassificationScaffold) -> bool:
    from dx_modelzoo.dataset import DATASET_REGISTRY

    return spec.dataset_name not in DATASET_REGISTRY


def _needs_custom_evaluator_template(spec: ClassificationScaffold) -> bool:
    from dx_modelzoo.evaluator import EVALUATOR_REGISTRY

    return spec.task not in EVALUATOR_REGISTRY


def _custom_postprocessing_types(spec: ClassificationScaffold) -> list[str]:
    from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY

    custom_types: list[str] = []
    for step in spec.postprocessing_steps:
        step_type = step.get("type")
        if not step_type:
            continue
        validated = validate_identifier(step_type, "Postprocessing type")
        if validated not in POSTPROCESSING_REGISTRY and validated not in custom_types:
            custom_types.append(validated)
    return custom_types


def _postprocessing_placeholder_class_name(step_type: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", step_type)
    normalized = "".join(part[:1].upper() + part[1:] for part in parts if part)
    return f"{normalized or 'Custom'}Postprocess"


def _render_custom_postprocessing_sections(spec: ClassificationScaffold) -> list[str]:
    sections: list[str] = []
    for step_type in _custom_postprocessing_types(spec):
        class_name = _postprocessing_placeholder_class_name(step_type)
        sections.append(
            textwrap.dedent(
                f"""\


                @POSTPROCESSING_REGISTRY.register("{step_type}")
                class {class_name}:
                    def __call__(self, outputs, **kwargs):
                        raise NotImplementedError("Implement __call__() for custom postprocessing type {step_type}")
                """
            ).rstrip()
        )
    return sections


def _append_missing_custom_postprocessing_sections(existing_text: str, spec: ClassificationScaffold) -> str:
    updated = existing_text
    for section, step_type in zip(
        _render_custom_postprocessing_sections(spec), _custom_postprocessing_types(spec), strict=False
    ):
        register_line = f'@POSTPROCESSING_REGISTRY.register("{step_type}")'
        if register_line in updated:
            continue
        if updated and not updated.endswith("\n"):
            updated += "\n"
        updated += section + "\n"
    return updated


def _render_custom_ops_template(spec: ClassificationScaffold) -> str:
    dataset_name = validate_identifier(spec.dataset_name, "Dataset name")
    evaluator_name = validate_identifier(spec.task, "Task")
    needs_dataset = _needs_custom_dataset_template(spec)
    needs_evaluator = _needs_custom_evaluator_template(spec)
    custom_post_sections = _render_custom_postprocessing_sections(spec)

    header = textwrap.dedent(
        """\
        \"\"\"Model-local custom registrations.

        Use this file as the single registration point for model-local dataset,
        evaluator, preprocessing, and postprocessing extensions.
        \"\"\"

        from dx_modelzoo.common.dataloader import DatasetBase
        from dx_modelzoo.dataset import DATASET_REGISTRY
        from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
        from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
        from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


        # Register model-local preprocessing/postprocessing ops here as needed.
        # Example:
        # @PREPROCESSING_REGISTRY.register("my_preprocess")
        # class MyPreprocess:
        #     def __call__(self, inputs):
        #         return inputs
        """
    )

    sections: list[str] = [header.rstrip()]
    if needs_dataset:
        sections.append(
            textwrap.dedent(
                f"""\


                @DATASET_REGISTRY.register("{dataset_name}")
                class CustomDataset(DatasetBase):
                    \"\"\"TODO: implement dataset loading for {dataset_name}.\"\"\"

                    def __len__(self) -> int:
                        raise NotImplementedError("Implement __len__() for {dataset_name}")

                    def __getitem__(self, idx: int):
                        raise NotImplementedError("Implement __getitem__() for {dataset_name}")
                """
            ).rstrip()
        )

    if needs_evaluator:
        sections.append(
            textwrap.dedent(
                f"""\


                @EVALUATOR_REGISTRY.register("{evaluator_name}")
                class CustomEvaluator(EvaluatorBase):
                    \"\"\"TODO: implement evaluator logic for {evaluator_name}.\"\"\"

                    def init_metrics(self):
                        raise NotImplementedError("Implement init_metrics() for {evaluator_name}")

                    def extract_inputs(self, batch_data):
                        raise NotImplementedError("Implement extract_inputs() for {evaluator_name}")

                    def process_batch_result(self, batch_data, output, metrics_state):
                        raise NotImplementedError("Implement process_batch_result() for {evaluator_name}")

                    def compute_final_metrics(self, metrics_state):
                        raise NotImplementedError("Implement compute_final_metrics() for {evaluator_name}")

                    def format_progress_desc(self, metrics_state, current_fps: float) -> str:
                        raise NotImplementedError("Implement format_progress_desc() for {evaluator_name}")
                """
            ).rstrip()
        )

    sections.extend(custom_post_sections)
    return "\n".join(sections) + "\n"


def _write_custom_ops_template(model_dir: Path, spec: ClassificationScaffold) -> None:
    if not (_needs_custom_dataset_template(spec) or _needs_custom_evaluator_template(spec)):
        return

    custom_ops_path = model_dir / "custom_ops.py"
    if custom_ops_path.exists():
        return

    custom_ops_text = _render_custom_ops_template(spec)
    temp_path = custom_ops_path.with_name(f"{custom_ops_path.name}.tmp")
    try:
        temp_path.write_text(custom_ops_text)
        temp_path.replace(custom_ops_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def ensure_no_duplicate_custom_name(custom_root: Path, model_name: str, target_path: Path) -> None:
    """Check that model_name doesn't already exist at a different path in custom_root.

    Args:
        custom_root: Root directory for custom models (e.g., ./custom)
        model_name: Name of the model to check
        target_path: The intended path for this model (allowed to match if same path)

    Raises:
        DuplicateCustomModelError: If model_name exists at a different path
    """
    for entry in scan_all_models(custom_root, source="custom"):
        if entry.name == model_name and entry.yaml_path != target_path:
            raise DuplicateCustomModelError(
                f"Duplicate custom model '{model_name}' already exists at {entry.yaml_path}"
            )


def write_classification_scaffold(
    custom_root: Path,
    spec: ClassificationScaffold,
    *,
    overwrite: bool = False,
) -> Path:
    custom_root = Path(custom_root)
    target_path = build_custom_model_path(
        custom_root,
        spec.family,
        spec.model_name,
        domain=spec.domain,
        task=spec.task,
    )
    ensure_no_duplicate_custom_name(custom_root, spec.model_name, target_path)

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists")

    yaml_text = render_classification_yaml(spec)
    needs_custom_ops = (
        _needs_custom_dataset_template(spec)
        or _needs_custom_evaluator_template(spec)
        or bool(_custom_postprocessing_types(spec))
    )
    custom_ops_path = target_path.parent / "custom_ops.py"
    should_create_custom_ops = needs_custom_ops and not custom_ops_path.exists()
    should_update_custom_ops = bool(_custom_postprocessing_types(spec)) and custom_ops_path.exists()
    custom_ops_text = _render_custom_ops_template(spec) if should_create_custom_ops else None
    updated_custom_ops_text = (
        _append_missing_custom_postprocessing_sections(custom_ops_path.read_text(), spec)
        if should_update_custom_ops
        else None
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")
    custom_ops_temp_path = custom_ops_path.with_name(f"{custom_ops_path.name}.tmp")
    backup_path = target_path.with_name(f"{target_path.name}.bak")
    published_custom_ops = False
    try:
        temp_path.write_text(yaml_text)
        if custom_ops_text is not None:
            custom_ops_temp_path.write_text(custom_ops_text)
        elif updated_custom_ops_text is not None:
            custom_ops_temp_path.write_text(updated_custom_ops_text)

        if overwrite and target_path.exists():
            backup_path.unlink(missing_ok=True)
            target_path.replace(backup_path)

        if custom_ops_text is not None or updated_custom_ops_text is not None:
            custom_ops_temp_path.replace(custom_ops_path)
            published_custom_ops = True

        temp_path.replace(target_path)
    except Exception:
        # Cleanup temp file if atomic replace fails
        temp_path.unlink(missing_ok=True)
        custom_ops_temp_path.unlink(missing_ok=True)
        if published_custom_ops:
            custom_ops_path.unlink(missing_ok=True)
        if backup_path.exists():
            backup_path.replace(target_path)
        raise
    backup_path.unlink(missing_ok=True)
    return target_path
