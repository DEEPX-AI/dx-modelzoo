from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from dx_modelzoo.common.exceptions import YamlValidationError
from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY
from dx_modelzoo.preprocessing import PREPROCESSING_REGISTRY


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML model config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise YamlValidationError(f"{path}: YAML root must be a mapping, got {type(config).__name__}")
    return config


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
                    f"{prefix}.layer[{i}] has invalid keys: {sorted(invalid)}. Allowed: {sorted(_valid_layer_keys)}"
                )

    elif ppu_type == 2:
        # Anchor-free multi-head: requires topk; layer is list
        if "topk" not in ppu_config:
            raise YamlValidationError(f"{prefix} type=2 requires 'topk'")
        if not isinstance(ppu_config["topk"], int):
            raise YamlValidationError(f"{prefix}.topk must be int")
        if not isinstance(layer, list):
            raise YamlValidationError(f"{prefix}.layer must be a list for type=2")
        _valid_layer_keys = {"bbox", "cls_conf", "mask_coeff"}
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
                    f"{prefix}.layer[{i}] has invalid keys: {sorted(invalid)}. Allowed: {sorted(_valid_layer_keys)}"
                )


def _validate_pre_optimize_config(pre_optimize: Any, yaml_path: str, profile_name: str) -> None:
    """Validate the pass-list shape accepted by dx_com.pre_optimize()."""
    prefix = f"{yaml_path}: profiles.{profile_name}.compile.pre_optimize"
    if not isinstance(pre_optimize, list):
        raise YamlValidationError(f"{prefix} must be a list")
    for i, entry in enumerate(pre_optimize):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise YamlValidationError(f"{prefix}[{i}] must be a single pass-name mapping")
        pass_name, pass_config = next(iter(entry.items()))
        if not isinstance(pass_name, str) or not pass_name.strip():
            raise YamlValidationError(f"{prefix}[{i}] pass name must be a non-empty string")
        if not isinstance(pass_config, dict):
            raise YamlValidationError(f"{prefix}[{i}].{pass_name} must be a mapping")


def _require_mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise YamlValidationError(f"{path} must be a mapping")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise YamlValidationError(f"{path} must be a non-empty string")
    return value


def _validate_inputs(
    inputs: Any,
    yaml_path: str,
    prefix: str = "inputs",
    symbolic_dimensions: Optional[set[str]] = None,
) -> None:
    if not isinstance(inputs, list) or not inputs:
        raise YamlValidationError(f"{yaml_path}: {prefix} must be a non-empty list")
    names = []
    symbolic_dimensions = symbolic_dimensions or set()
    for i, inp in enumerate(inputs):
        inp = _require_mapping(inp, f"{yaml_path}: {prefix}[{i}]")
        name = _require_string(inp.get("name"), f"{yaml_path}: {prefix}[{i}].name")
        if name in names:
            raise YamlValidationError(f"{yaml_path}: {prefix}[{i}].name must be unique; duplicate '{name}'")
        names.append(name)
        shape = inp.get("shape")
        if "shape" not in inp:
            raise YamlValidationError(f"{yaml_path}: {prefix}[{i}] missing 'shape'")
        if not isinstance(shape, list) or not shape:
            raise YamlValidationError(f"{yaml_path}: {prefix}[{i}].shape must be a non-empty list")
        input_symbols = set(symbolic_dimensions)
        for key in ("symbolic_dimensions", "symbolic_dims", "symbols"):
            value = inp.get(key, {})
            if isinstance(value, dict):
                input_symbols.update(value)
            elif isinstance(value, list):
                input_symbols.update(item for item in value if isinstance(item, str))
        for j, dimension in enumerate(shape):
            if isinstance(dimension, bool) or (
                not isinstance(dimension, int) and not (isinstance(dimension, str) and dimension.strip())
            ):
                raise YamlValidationError(
                    f"{yaml_path}: {prefix}[{i}].shape[{j}] must be a positive integer or symbolic dimension"
                )
            if isinstance(dimension, int) and dimension <= 0:
                raise YamlValidationError(f"{yaml_path}: {prefix}[{i}].shape[{j}] must be a positive integer")
            if isinstance(dimension, str) and dimension not in input_symbols:
                raise YamlValidationError(
                    f"{yaml_path}: {prefix}[{i}].shape[{j}] symbolic dimension {dimension!r} must be explicitly defined"
                )
        if "dtype" in inp:
            _require_string(inp["dtype"], f"{yaml_path}: {prefix}[{i}].dtype")
        if "layout" in inp:
            _require_string(inp["layout"], f"{yaml_path}: {prefix}[{i}].layout")


def _validate_preprocessing(
    preprocessing: Any,
    yaml_path: str,
    prefix: str = "preprocessing",
    input_names: Optional[set[str]] = None,
) -> None:
    if isinstance(preprocessing, list):
        if input_names is not None and len(input_names) > 1:
            raise YamlValidationError(
                f"{yaml_path}: {prefix} must be a mapping keyed by input names for multi-input models"
            )
        modal_steps = {prefix: preprocessing}
    elif isinstance(preprocessing, dict):
        modal_steps = preprocessing
        if not modal_steps or any(not isinstance(steps, list) for steps in modal_steps.values()):
            raise YamlValidationError(f"{yaml_path}: {prefix} must contain lists of steps")
        if input_names is not None and set(modal_steps) != input_names:
            raise YamlValidationError(
                f"{yaml_path}: {prefix} keys must match input names. "
                f"Missing: {sorted(input_names - set(modal_steps))}; "
                f"unknown: {sorted(set(modal_steps) - input_names)}"
            )
    elif input_names is not None and len(input_names) > 1:
        raise YamlValidationError(
            f"{yaml_path}: {prefix} must be a mapping keyed by input names for multi-input models"
        )
    else:
        raise YamlValidationError(f"{yaml_path}: {prefix} must be a list or mapping")

    for modal, steps in modal_steps.items():
        for i, step in enumerate(steps):
            step = _require_mapping(step, f"{yaml_path}: {modal}[{i}]")
            step_type = _require_string(step.get("type"), f"{yaml_path}: {modal}[{i}].type")
            if step_type not in PREPROCESSING_REGISTRY:
                raise YamlValidationError(
                    f"{yaml_path}: Unknown preprocessing type '{step_type}'. "
                    f"Available: {sorted(PREPROCESSING_REGISTRY.list())}"
                )


def _validate_postprocessing(postprocessing: Any, yaml_path: str) -> None:
    if not isinstance(postprocessing, list):
        raise YamlValidationError(f"{yaml_path}: postprocessing must be a list")
    for i, step in enumerate(postprocessing):
        step = _require_mapping(step, f"{yaml_path}: postprocessing[{i}]")
        _require_string(step.get("type"), f"{yaml_path}: postprocessing[{i}].type")


def _validate_dataset_and_artifacts(config: Dict[str, Any], yaml_path: str, prefix: str = "") -> None:
    dataset_path = f"{prefix}dataset" if prefix else "dataset"
    dataset = _require_mapping(config.get("dataset"), f"{yaml_path}: {dataset_path}")
    _require_string(dataset.get("type"), f"{yaml_path}: {dataset_path}.type")
    if "eval_path" in dataset:
        _require_string(dataset["eval_path"], f"{yaml_path}: {dataset_path}.eval_path")

    artifacts_path = f"{prefix}artifacts" if prefix else "artifacts"
    artifacts = _require_mapping(config.get("artifacts"), f"{yaml_path}: {artifacts_path}")
    _require_string(artifacts.get("path"), f"{yaml_path}: {artifacts_path}.path")


def _validate_profiles(profiles: Any, yaml_path: str) -> None:
    if not isinstance(profiles, dict) or not profiles:
        raise YamlValidationError(f"{yaml_path}: At least one profile required")
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name.strip():
            raise YamlValidationError(f"{yaml_path}: profile names must be non-empty strings")
        profile = _require_mapping(profile, f"{yaml_path}: profiles.{name}")
        target = _require_string(profile.get("target"), f"{yaml_path}: profiles.{name}.target")
        if target not in {"onnx", "dxnn"}:
            raise YamlValidationError(f"{yaml_path}: profiles.{name}.target must be one of: onnx, dxnn")

        if "runtime" not in profile:
            raise YamlValidationError(f"{yaml_path}: profiles.{name} missing 'runtime' section")
        runtime = _require_mapping(profile["runtime"], f"{yaml_path}: profiles.{name}.runtime")
        if "device" not in runtime or runtime["device"] is None:
            raise YamlValidationError(f"{yaml_path}: profiles.{name}.runtime missing 'device'")
        if isinstance(runtime["device"], str) and not runtime["device"].strip():
            raise YamlValidationError(f"{yaml_path}: profiles.{name}.runtime.device must not be empty")
        if "batch_size" in runtime and (
            isinstance(runtime["batch_size"], bool)
            or not isinstance(runtime["batch_size"], int)
            or runtime["batch_size"] <= 0
        ):
            raise YamlValidationError(f"{yaml_path}: profiles.{name}.runtime.batch_size must be a positive integer")
        if "start_method" in runtime and (
            not isinstance(runtime["start_method"], str)
            or runtime["start_method"] not in {"auto", "forkserver", "spawn"}
        ):
            raise YamlValidationError(
                f"{yaml_path}: profiles.{name}.runtime.start_method must be one of: auto, forkserver, spawn"
            )


def validate_yaml(config: Dict[str, Any], yaml_path: str) -> None:
    """Validate YAML structure. Raises YamlValidationError on problems."""

    # Pipeline YAML — per-stage validation
    if "pipeline" in config:
        return _validate_pipeline_yaml(config, yaml_path)

    # 1. Required fields
    required = ["name", "task", "inputs", "preprocessing", "dataset", "artifacts", "profiles"]
    missing = [f for f in required if f not in config]
    if missing:
        raise YamlValidationError(f"{yaml_path}: Missing required fields: {missing}")
    _require_string(config["name"], f"{yaml_path}: name")
    _require_string(config["task"], f"{yaml_path}: task")

    # 2. Inputs structure
    symbolic_dimensions = set()
    for key in ("symbolic_dimensions", "symbolic_dims", "symbols"):
        value = config.get(key, {})
        if isinstance(value, dict):
            symbolic_dimensions.update(value)
        elif isinstance(value, list):
            symbolic_dimensions.update(item for item in value if isinstance(item, str))
    input_names = {item.get("name") for item in config["inputs"] if isinstance(item, dict)}
    _validate_inputs(config["inputs"], yaml_path, symbolic_dimensions=symbolic_dimensions)

    # 3. Preprocessing type validation
    _validate_preprocessing(config["preprocessing"], yaml_path, input_names=input_names)

    # 4. Postprocessing type validation (optional field, warn for unknown)
    if "postprocessing" in config and config["postprocessing"] is not None:
        _validate_postprocessing(config["postprocessing"], yaml_path)
        for step in config["postprocessing"]:
            if step.get("type") and step["type"] not in POSTPROCESSING_REGISTRY:
                warnings.warn(
                    f"{yaml_path}: Unknown postprocessing type '{step['type']}'. "
                    "Ensure custom_ops.py or additional registration is provided."
                )

    # 4b. Validate preprocessing steps in appendix.label_preprocessing and hr_preprocessing
    appendix = config.get("appendix", {})
    if isinstance(appendix, dict):
        for preproc_key in ("label_preprocessing", "hr_preprocessing"):
            if preproc_key in appendix:
                _validate_preprocessing(appendix[preproc_key], yaml_path, f"appendix.{preproc_key}")

    # 5. Profiles structure
    _validate_dataset_and_artifacts(config, yaml_path)
    _validate_profiles(config["profiles"], yaml_path)

    for name, profile in config["profiles"].items():
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

        if profile["target"] == "dxnn":
            _require_mapping(profile.get("compile"), f"{yaml_path}: profiles.{name}.compile")
        elif "compile" in profile:
            _require_mapping(profile["compile"], f"{yaml_path}: profiles.{name}.compile")

        # 5b. calibration structure validation
        if "compile" in profile and "calibration" in profile["compile"]:
            calibration = profile["compile"]["calibration"]
            if not isinstance(calibration, dict):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.compile.calibration must be a dict")

            if "method" in calibration and not isinstance(calibration["method"], str):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.compile.calibration.method must be str")
            if "num" in calibration and (
                not isinstance(calibration["num"], int) or isinstance(calibration["num"], bool)
            ):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.compile.calibration.num must be int")
            if "seed" in calibration and (
                not isinstance(calibration["seed"], int) or isinstance(calibration["seed"], bool)
            ):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.compile.calibration.seed must be int")
            if "use_q_pro" in calibration and not isinstance(calibration["use_q_pro"], bool):
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.calibration.use_q_pro must be a boolean"
                )
            if "wbit" in calibration and (
                not isinstance(calibration["wbit"], int) or isinstance(calibration["wbit"], bool)
            ):
                raise YamlValidationError(f"{yaml_path}: profiles.{name}.compile.calibration.wbit must be int")
            if "enhanced_scheme" in calibration and not isinstance(calibration["enhanced_scheme"], dict):
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.calibration.enhanced_scheme must be a dict"
                )

        # 5c. ppu_config structure validation
        if "compile" in profile and "ppu_config" in profile["compile"]:
            if profile.get("target") != "dxnn":
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.ppu_config is only allowed for target=dxnn"
                )
            _validate_ppu_config(profile["compile"]["ppu_config"], yaml_path, name)
        if "compile" in profile and "pre_optimize" in profile["compile"]:
            if profile.get("target") != "dxnn":
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile.pre_optimize is only allowed for target=dxnn"
                )
            if "ppu_config" in profile["compile"]:
                raise YamlValidationError(
                    f"{yaml_path}: profiles.{name}.compile cannot contain both 'ppu_config' and 'pre_optimize'"
                )
            _validate_pre_optimize_config(profile["compile"]["pre_optimize"], yaml_path, name)


def _validate_pipeline_yaml(config: Dict[str, Any], yaml_path: str) -> None:
    """Validate a pipeline YAML with per-stage definitions."""
    required_top = ["name", "pipeline", "profiles", "dataset"]
    missing = [f for f in required_top if f not in config]
    if missing:
        raise YamlValidationError(f"{yaml_path}: Pipeline YAML missing required fields: {missing}")
    _require_string(config["name"], f"{yaml_path}: name")

    pipeline = config["pipeline"]
    if not isinstance(pipeline, dict) or not pipeline:
        raise YamlValidationError(f"{yaml_path}: 'pipeline' must be a non-empty mapping of stages")

    global_symbols = set()
    for key in ("symbolic_dimensions", "symbolic_dims", "symbols"):
        value = config.get(key, {})
        if isinstance(value, dict):
            global_symbols.update(value)
        elif isinstance(value, list):
            global_symbols.update(item for item in value if isinstance(item, str))

    for stage_name, stage_cfg in pipeline.items():
        _require_string(stage_name, f"{yaml_path}: pipeline stage name")
        prefix = f"{yaml_path}: pipeline.{stage_name}"
        stage_cfg = _require_mapping(stage_cfg, prefix)
        stage_required = ["inputs", "preprocessing", "artifacts"]
        missing = [f for f in stage_required if f not in stage_cfg]
        if missing:
            raise YamlValidationError(f"{prefix}: Missing required fields: {missing}")

        stage_symbols = set(global_symbols)
        for key in ("symbolic_dimensions", "symbolic_dims", "symbols"):
            value = stage_cfg.get(key, {})
            if isinstance(value, dict):
                stage_symbols.update(value)
            elif isinstance(value, list):
                stage_symbols.update(item for item in value if isinstance(item, str))
        stage_input_names = {item.get("name") for item in stage_cfg["inputs"] if isinstance(item, dict)}
        _validate_inputs(
            stage_cfg["inputs"],
            yaml_path,
            f"pipeline.{stage_name}.inputs",
            symbolic_dimensions=stage_symbols,
        )
        _validate_preprocessing(
            stage_cfg["preprocessing"],
            yaml_path,
            f"pipeline.{stage_name}.preprocessing",
            input_names=stage_input_names,
        )
        artifacts = _require_mapping(stage_cfg["artifacts"], f"{prefix}.artifacts")
        _require_string(artifacts.get("path"), f"{prefix}.artifacts.path")

    # Validate shared top-level profiles
    profiles = config["profiles"]
    _validate_profiles(profiles, yaml_path)
    for pname, profile in profiles.items():
        if profile["target"] == "dxnn":
            _require_mapping(profile.get("compile"), f"{yaml_path}: profiles.{pname}.compile")
        elif "compile" in profile:
            _require_mapping(profile["compile"], f"{yaml_path}: profiles.{pname}.compile")
        # Validate compile section for NPU profiles (same as non-pipeline)
        target = profile.get("target", "onnx")
        if target != "onnx":
            compile_cfg = profile.get("compile", {})
            if not compile_cfg:
                warnings.warn(f"{yaml_path}: profiles.{pname} targets NPU but has no 'compile' section.")
            if "ppu_config" in compile_cfg:
                _validate_ppu_config(compile_cfg["ppu_config"], yaml_path, pname)
            if "pre_optimize" in compile_cfg:
                if "ppu_config" in compile_cfg:
                    raise YamlValidationError(
                        f"{yaml_path}: profiles.{pname}.compile cannot contain both 'ppu_config' and 'pre_optimize'"
                    )
                _validate_pre_optimize_config(compile_cfg["pre_optimize"], yaml_path, pname)
