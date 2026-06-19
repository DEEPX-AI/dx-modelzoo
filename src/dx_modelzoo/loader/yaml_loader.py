from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from dx_modelzoo.common.exceptions import YamlValidationError
from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML model config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise YamlValidationError(f"{path}: YAML root must be a mapping, got {type(config).__name__}")
    return config


def _flatten_preprocessing(preprocessing: Any) -> List[Dict[str, Any]]:
    """Flatten preprocessing config (list or multimodal dict) into list of step dicts."""
    if isinstance(preprocessing, list):
        return preprocessing
    elif isinstance(preprocessing, dict):
        steps = []
        for modal_steps in preprocessing.values():
            if isinstance(modal_steps, list):
                steps.extend(modal_steps)
        return steps
    return []


def _validate_ppu_config(ppu_config: Any, yaml_path: str, profile_name: str) -> None:
    """Validate ppu_config structure based on type."""
    prefix = f"{yaml_path}: profiles.{profile_name}.compile.ppu_config"

    if not isinstance(ppu_config, dict):
        raise YamlValidationError(f"{prefix} must be a dict")

    # Common required keys
    for key in ("type", "num_classes", "layer"):
        if key not in ppu_config:
            raise YamlValidationError(f"{prefix} missing required key '{key}'")

    ppu_type = ppu_config["type"]
    if not isinstance(ppu_type, int) or ppu_type not in (0, 1, 2):
        raise YamlValidationError(f"{prefix}.type must be 0, 1, or 2")

    if not isinstance(ppu_config["num_classes"], int):
        raise YamlValidationError(f"{prefix}.num_classes must be int")

    layer = ppu_config["layer"]

    if ppu_type == 0:
        # Anchor-based: requires conf_thres, activation; layer is dict
        for key in ("conf_thres", "activation"):
            if key not in ppu_config:
                raise YamlValidationError(f"{prefix} type=0 requires '{key}'")
        if not isinstance(ppu_config["activation"], str):
            raise YamlValidationError(f"{prefix}.activation must be str")
        if not isinstance(layer, dict):
            raise YamlValidationError(f"{prefix}.layer must be a dict for type=0")
        for layer_name, layer_val in layer.items():
            if not isinstance(layer_val, dict):
                raise YamlValidationError(f"{prefix}.layer.{layer_name} must be a dict")
            if "num_anchors" not in layer_val:
                raise YamlValidationError(f"{prefix}.layer.{layer_name} missing 'num_anchors'")
            if not isinstance(layer_val["num_anchors"], int):
                raise YamlValidationError(f"{prefix}.layer.{layer_name}.num_anchors must be int")

    elif ppu_type == 1:
        # Anchor-free single-head: requires conf_thres; layer is list
        if "conf_thres" not in ppu_config:
            raise YamlValidationError(f"{prefix} type=1 requires 'conf_thres'")
        if not isinstance(layer, list):
            raise YamlValidationError(f"{prefix}.layer must be a list for type=1")
        _valid_layer_keys = {"bbox", "cls_conf", "obj_conf"}
        for i, entry in enumerate(layer):
            if not isinstance(entry, dict):
                raise YamlValidationError(f"{prefix}.layer[{i}] must be a dict")
            for req_key in ("bbox", "cls_conf"):
                if req_key not in entry:
                    raise YamlValidationError(f"{prefix}.layer[{i}] missing '{req_key}'")
                if not isinstance(entry[req_key], str):
                    raise YamlValidationError(f"{prefix}.layer[{i}].{req_key} must be str")
            invalid = set(entry.keys()) - _valid_layer_keys
            if invalid:
                raise YamlValidationError(
                    f"{prefix}.layer[{i}] has invalid keys: {sorted(invalid)}. " f"Allowed: {sorted(_valid_layer_keys)}"
                )

    elif ppu_type == 2:
        # Anchor-free multi-head: requires topk; layer is list
        if "topk" not in ppu_config:
            raise YamlValidationError(f"{prefix} type=2 requires 'topk'")
        if not isinstance(ppu_config["topk"], int):
            raise YamlValidationError(f"{prefix}.topk must be int")
        if not isinstance(layer, list):
            raise YamlValidationError(f"{prefix}.layer must be a list for type=2")
        _valid_layer_keys = {"bbox", "cls_conf"}
        for i, entry in enumerate(layer):
            if not isinstance(entry, dict):
                raise YamlValidationError(f"{prefix}.layer[{i}] must be a dict")
            for req_key in ("bbox", "cls_conf"):
                if req_key not in entry:
                    raise YamlValidationError(f"{prefix}.layer[{i}] missing '{req_key}'")
                if not isinstance(entry[req_key], str):
                    raise YamlValidationError(f"{prefix}.layer[{i}].{req_key} must be str")
            invalid = set(entry.keys()) - _valid_layer_keys
            if invalid:
                raise YamlValidationError(
                    f"{prefix}.layer[{i}] has invalid keys: {sorted(invalid)}. " f"Allowed: {sorted(_valid_layer_keys)}"
                )


def validate_yaml(config: Dict[str, Any], yaml_path: str) -> None:
    """Validate YAML structure. Raises YamlValidationError on problems."""

    # Pipeline YAML — per-stage validation
    if "pipeline" in config:
        return _validate_pipeline_yaml(config, yaml_path)

    # 1. Required fields
    required = ["name", "inputs", "preprocessing", "dataset", "artifacts", "profiles"]
    missing = [f for f in required if f not in config]
    if missing:
        raise YamlValidationError(f"{yaml_path}: Missing required fields: {missing}")

    # 2. Inputs structure
    for i, inp in enumerate(config["inputs"]):
        if "shape" not in inp:
            raise YamlValidationError(f"{yaml_path}: inputs[{i}] missing 'shape'")

    # 3. Preprocessing type validation
    steps = _flatten_preprocessing(config["preprocessing"])
    for step in steps:
        if "type" not in step:
            raise YamlValidationError(f"{yaml_path}: preprocessing step missing 'type': {step}")
        if step["type"] not in PREPROCESSING_REGISTRY:
            raise YamlValidationError(
                f"{yaml_path}: Unknown preprocessing type '{step['type']}'. "
                f"Available: {sorted(PREPROCESSING_REGISTRY.list())}"
            )

    # 4. Postprocessing type validation (optional field, warn for unknown)
    if "postprocessing" in config and config["postprocessing"]:
        for step in config["postprocessing"]:
            if step.get("type") and step["type"] not in POSTPROCESSING_REGISTRY:
                warnings.warn(
                    f"{yaml_path}: Unknown postprocessing type '{step['type']}'. "
                    f"Ensure custom_ops.py or additional registration is provided."
                )

    # 4b. Validate preprocessing steps in appendix.label_preprocessing and hr_preprocessing
    appendix = config.get("appendix", {})
    if isinstance(appendix, dict):
        for preproc_key in ("label_preprocessing", "hr_preprocessing"):
            preproc_steps = appendix.get(preproc_key, [])
            if isinstance(preproc_steps, list):
                for step in preproc_steps:
                    if "type" not in step:
                        raise YamlValidationError(f"{yaml_path}: appendix.{preproc_key} step missing 'type': {step}")
                    if step["type"] not in PREPROCESSING_REGISTRY:
                        raise YamlValidationError(
                            f"{yaml_path}: Unknown appendix.{preproc_key} type '{step['type']}'. "
                            f"Available: {sorted(PREPROCESSING_REGISTRY.list())}"
                        )

    # 5. Profiles structure
    if not config["profiles"]:
        raise YamlValidationError(f"{yaml_path}: At least one profile required")

    for name, profile in config["profiles"].items():
        if "runtime" not in profile:
            raise YamlValidationError(f"{yaml_path}: profiles.{name} missing 'runtime' section")
        if "device" not in profile["runtime"]:
            raise YamlValidationError(f"{yaml_path}: profiles.{name}.runtime missing 'device'")
        if profile["runtime"]["device"] == "npu" and "compile" not in profile:
            warnings.warn(f"{yaml_path}: profiles.{name} targets NPU but has no 'compile' section.")

        # 5a. runtime.async validation
        if "async" in profile["runtime"]:
            if not isinstance(profile["runtime"]["async"], bool):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.runtime.async must be a boolean")

        # 5a-2. runtime.buffer_count validation (engine I/O buffer count, 1..100)
        if "buffer_count" in profile["runtime"]:
            buffer_count = profile["runtime"]["buffer_count"]
            if not isinstance(buffer_count, int) or isinstance(buffer_count, bool) or not (1 <= buffer_count <= 100):
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.runtime.buffer_count must be an integer in [1, 100]"
                )

        # 5a-3. runtime.use_ort validation
        if "use_ort" in profile["runtime"]:
            if not isinstance(profile["runtime"]["use_ort"], bool):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.runtime.use_ort must be a boolean")

        # 5b. quantization structure validation
        if "compile" in profile and "quantization" in profile["compile"]:
            quant = profile["compile"]["quantization"]
            if not isinstance(quant, dict):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.compile.quantization must be a dict")

            valid_keys = {"lite", "master", "pro", "p0", "p1", "p2", "p3", "p4", "p5"}
            invalid_keys = set(quant.keys()) - valid_keys
            if invalid_keys:
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.quantization has invalid keys: "
                    f"{sorted(invalid_keys)}. Allowed: {sorted(valid_keys)}"
                )

            for qkey in ("lite", "master", "pro"):
                if qkey in quant:
                    qval = quant[qkey]
                    if qval is not None and not isinstance(qval, dict):
                        raise YamlValidationError(
                            f"{yaml_path}: profiles.{name}.compile.quantization.{qkey} must be a dict or empty"
                        )
                    if isinstance(qval, dict):
                        if "num_samples" in qval and not isinstance(qval["num_samples"], int):
                            raise YamlValidationError(
                                f"{yaml_path}: profiles.{name}.compile.quantization.{qkey}.num_samples must be int"
                            )
                        if "method" in qval and not isinstance(qval["method"], str):
                            raise YamlValidationError(
                                f"{yaml_path}: profiles.{name}.compile.quantization.{qkey}.method must be str"
                            )

            mode_keys = {"lite", "master", "pro"} & set(quant.keys())
            p_keys = sorted(k for k in quant if k in {"p0", "p1", "p2", "p3", "p4", "p5"})
            if len(mode_keys) > 1:
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.quantization cannot mix "
                    f"modes {sorted(mode_keys)}"
                )
            if mode_keys and p_keys:
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.quantization cannot mix "
                    f"{sorted(mode_keys)} with p-levels {p_keys}"
                )

            if p_keys:
                valid_p_combinations = {
                    "p0",
                    "p1",
                    "p2",
                    "p3",
                    "p4",
                    "p5",
                    "p1p3",
                    "p2p3",
                    "p1p4",
                    "p2p4",
                    "p1p5",
                    "p2p5",
                }
                combo = "".join(p_keys)
                if combo not in valid_p_combinations:
                    raise YamlValidationError(
                        f"{yaml_path}: profiles.{name}.compile.quantization has invalid "
                        f"p-level combination: {p_keys}. "
                        f"Valid combinations: {sorted(valid_p_combinations)}"
                    )

        # 5c. ppu_config structure validation
        if "compile" in profile and "ppu_config" in profile["compile"]:
            if profile.get("target") != "dxnn":
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.ppu_config is only allowed for target=dxnn"
                )
            _validate_ppu_config(profile["compile"]["ppu_config"], yaml_path, name)


def _validate_pipeline_yaml(config: Dict[str, Any], yaml_path: str) -> None:
    """Validate a pipeline YAML with per-stage definitions."""
    required_top = ["name", "pipeline", "profiles", "dataset"]
    missing = [f for f in required_top if f not in config]
    if missing:
        raise YamlValidationError(f"{yaml_path}: Pipeline YAML missing required fields: {missing}")

    pipeline = config["pipeline"]
    if not isinstance(pipeline, dict) or not pipeline:
        raise YamlValidationError(f"{yaml_path}: 'pipeline' must be a non-empty mapping of stages")

    for stage_name, stage_cfg in pipeline.items():
        prefix = f"{yaml_path}: pipeline.{stage_name}"
        stage_required = ["inputs", "preprocessing", "artifacts"]
        missing = [f for f in stage_required if f not in stage_cfg]
        if missing:
            raise YamlValidationError(f"{prefix}: Missing required fields: {missing}")

        for i, inp in enumerate(stage_cfg["inputs"]):
            if "shape" not in inp:
                raise YamlValidationError(f"{prefix}: inputs[{i}] missing 'shape'")

        steps = _flatten_preprocessing(stage_cfg["preprocessing"])
        for step in steps:
            if "type" not in step:
                raise YamlValidationError(f"{prefix}: preprocessing step missing 'type': {step}")
            if step["type"] not in PREPROCESSING_REGISTRY:
                raise YamlValidationError(
                    f"{prefix}: Unknown preprocessing type '{step['type']}'. "
                    f"Available: {sorted(PREPROCESSING_REGISTRY.list())}"
                )

    # Validate shared top-level profiles
    profiles = config["profiles"]
    if not profiles:
        raise YamlValidationError(f"{yaml_path}: At least one profile required")
    for pname, profile in profiles.items():
        if "runtime" not in profile:
            raise YamlValidationError(f"{yaml_path}: profiles.{pname} missing 'runtime' section")
        if "device" not in profile["runtime"]:
            raise YamlValidationError(f"{yaml_path}: profiles.{pname}.runtime missing 'device'")
        # Validate compile section for NPU profiles (same as non-pipeline)
        target = profile.get("target", "onnx")
        if target != "onnx":
            compile_cfg = profile.get("compile", {})
            if not compile_cfg:
                warnings.warn(f"{yaml_path}: profiles.{pname} targets NPU but has no 'compile' section.")
            if "ppu_config" in compile_cfg:
                _validate_ppu_config(compile_cfg["ppu_config"], yaml_path, pname)
