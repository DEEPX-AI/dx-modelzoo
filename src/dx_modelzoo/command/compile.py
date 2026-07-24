from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from dx_modelzoo.loader.model_builder import ModelBuilder
from dx_modelzoo.preprocessing import PreprocessingPipeline


def _remove_expanddim(preprocessing_config):
    """Remove expanddim steps from preprocessing config.

    Handles both single-input (list) and multi-input (dict) configs.
    DataLoader already adds the batch dimension, so expanddim is unnecessary.
    """
    if isinstance(preprocessing_config, list):
        return [step for step in preprocessing_config if step.get("type") != "expanddim"]
    if isinstance(preprocessing_config, dict):
        return {
            name: [step for step in steps if step.get("type") != "expanddim"]
            for name, steps in preprocessing_config.items()
            if isinstance(steps, list)
        }
    return preprocessing_config


def compile_model(
    builder: ModelBuilder,
    profile_name: str,
    output: Optional[str] = None,
    model_path: Optional[str] = None,
    use_gpu: bool = False,
    debug: bool = False,
    stage: Optional[str] = None,
) -> None:
    """Compile a model using dx_com.

    Reads compile settings from the YAML profile and calls dx_com.compile().
    For pipeline YAMLs, pass ``stage`` to compile a specific pipeline stage.
    """
    profile = builder.get_profile(profile_name)
    compile_config = profile.get("compile")
    if compile_config is None:
        raise ValueError(
            f"Profile '{profile_name}' has no compile section. " "Only profiles with compile config can be compiled."
        )

    try:
        from dx_com import compile as dx_compile
    except ImportError:
        raise ImportError("dx_com is not installed. " "Install it to use compile: pip install dx_com")

    # Resolve model path
    if model_path is None:
        if stage and builder.is_pipeline:
            artifacts = builder.get_stage_config(stage).get("artifacts", {})
        else:
            artifacts = builder.config.get("artifacts", {})
        base_path = artifacts.get("path") or artifacts.get("onnx")
        if base_path:
            base_path = Path(base_path)
            if base_path.suffix == ".onnx":
                model_path = str(base_path)
            else:
                model_path = str(base_path / f"{builder.name}.onnx")
    if model_path is None:
        stage_hint = f" (stage '{stage}')" if stage else ""
        raise ValueError(f"No model path specified{stage_hint} (--model-path or artifacts.path)")

    onnx_path = Path(model_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if output is None:
        output_dir = onnx_path.parent
        output_file = output_dir / f"{onnx_path.stem}_{profile_name}.dxnn"
    else:
        output = Path(output)
        if output.suffix == ".dxnn":
            output_dir = output.parent
            output_file = output
        else:
            output_dir = output
            output_file = output_dir / f"{onnx_path.stem}_{profile_name}.dxnn"
        output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_file = str(output_dir / f"{onnx_path.stem}.dxnn")

    opt_level = compile_config.get("opt_level", 1)

    quantization_config = compile_config.get("quantization", {})
    if not isinstance(quantization_config, dict):
        raise ValueError(f"'quantization' must be a dict, got: {type(quantization_config).__name__}")

    _VALID_P_LEVELS = {"p0", "p1", "p2", "p3", "p4", "p5"}
    _VALID_P_COMBINATIONS = {
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

    # Determine quantization mode from top-level keys
    quant_keys = set(quantization_config.keys())

    enhanced_scheme = None
    use_q_pro = False
    if "pro" in quant_keys:
        if quant_keys != {"pro"}:
            raise ValueError(f"'pro' cannot be combined with other quantization keys, got: {sorted(quant_keys)}.")
        pro_opts = quantization_config["pro"] or {}
        use_q_pro = True
        calibration_method = pro_opts.get("method", "ema")
        calibration_num = pro_opts.get("num_samples", 100)
    elif "lite" in quant_keys:
        lite_opts = quantization_config["lite"] or {}
        calibration_method = lite_opts.get("method", "ema")
        calibration_num = lite_opts.get("num_samples", 100)
    elif "master" in quant_keys:
        master_opts = quantization_config["master"] or {}
        calibration_method = master_opts.get("method", "ema")
        calibration_num = master_opts.get("num_samples", 100)
    elif quant_keys and quant_keys <= _VALID_P_LEVELS:
        # Validate the combination of p-levels
        combo_key = "".join(sorted(quant_keys, key=lambda k: int(k[1:])))
        if combo_key not in _VALID_P_COMBINATIONS:
            raise ValueError(
                f"Unsupported p-level combination: {sorted(quant_keys)}. "
                f"Valid combinations: {sorted(_VALID_P_COMBINATIONS)}"
            )
        enhanced_scheme = {}
        calibration_method = "ema"
        calibration_num = 100
        for level in sorted(quant_keys, key=lambda k: int(k[1:])):
            key = f"DXQ-{level.upper()}"
            per_level_opts = quantization_config[level]
            if per_level_opts and isinstance(per_level_opts, dict):
                enhanced_scheme[key] = per_level_opts
                calibration_method = per_level_opts.get("method", calibration_method)
                calibration_num = per_level_opts.get("num_samples", calibration_num)
            else:
                enhanced_scheme[key] = True
    else:
        raise ValueError(
            f"Unsupported quantization keys: {sorted(quant_keys)}. "
            f"Expected 'pro', 'lite', 'master', or p-levels (p0-p5)."
        )

    # Build calibration dataloader with full preprocessing (ONNX input format)
    # Exclude expanddim — DataLoader already adds batch dimension
    dataset = builder.build_dataset()
    if stage and builder.is_pipeline:
        preproc_config = _remove_expanddim(builder.config["pipeline"][stage]["preprocessing"])
        input_specs = builder.config["pipeline"][stage].get("inputs", [])
    else:
        preproc_config = _remove_expanddim(builder.config["preprocessing"])
        input_specs = builder.config.get("inputs", [])
    dataset.preprocessing = PreprocessingPipeline(preproc_config, npu_mode=False)
    dataloader = _make_calibration_loader(dataset, calibration_num, input_specs)

    # Run compilation
    ppu_config_dict = compile_config.get("ppu_config")

    compile_params = dict(
        model=str(onnx_path),
        # output=str(output_dir),
        output_dir=str(output_dir),
        config=None,
        dataloader=dataloader,
        # use_gpu=use_gpu,
        quantization_device="cuda" if use_gpu else "cpu",
        opt_level=0 if debug else opt_level,
        aggressive_partitioning=False,
        # save_log=debug,
        gen_log=debug,
        ###
        # release=True,
        calibration_method=calibration_method,
        calibration_num=calibration_num,
        enhanced_scheme=enhanced_scheme,
    )

    if use_q_pro:
        compile_params["use_q_pro"] = True
    if ppu_config_dict:
        from dx_com import PPUConfig

        ppu_config = PPUConfig.parse(ppu_config_dict)
        compile_params["ppu_config"] = ppu_config

    dx_compile(**compile_params)

    # dx_com outputs {model_name}.dxnn — rename to expected output name if different
    raw_output = Path(raw_output_file)
    if raw_output != output_file and raw_output.exists():
        shutil.move(str(raw_output), str(output_file))

    logger.success(f"Compilation complete. Output: {output_dir}")


def _make_calibration_loader(dataset, num_samples: int, input_specs=None):
    """Create a calibration dataloader that yields torch.Tensor batches.

    dx_com expects a DataLoader yielding torch.Tensor elements.
    Our datasets return (input, label, ...) tuples with numpy arrays,
    so we wrap with a dataset that extracts the input and converts to tensor.

    For multi-input models (e.g. BlazeFace with conf_threshold, iou_threshold),
    constant inputs (those with a ``value`` field in YAML) are included as
    additional tensors so the compiler receives the expected number of inputs.
    """
    import numpy as np
    import torch

    # Identify constant inputs (have 'value' in spec) — skip the first (variable) input
    constant_inputs = {}  # name → tensor
    primary_input_name = input_specs[0]["name"] if input_specs else "input"
    if input_specs and len(input_specs) > 1:
        for spec in input_specs[1:]:
            if "value" in spec:
                dtype_str = spec.get("dtype", "float32")
                dtype_map = {
                    "float32": (np.float32, torch.float32),
                    "float16": (np.float16, torch.float16),
                    "int32": (np.int32, torch.int32),
                    "int64": (np.int64, torch.int64),
                }
                np_dtype, torch_dtype = dtype_map.get(dtype_str, (np.float32, torch.float32))
                shape = spec["shape"]
                # Remove batch dim — DataLoader adds it back via collation
                if shape:
                    shape = shape[1:]  # [1] → [], [1,3,H,W] → [3,H,W]
                val = np.full(shape if shape else (), spec["value"], dtype=np_dtype)
                constant_inputs[spec["name"]] = torch.from_numpy(val).to(torch_dtype)

    class _CalibrationDataset:
        def __init__(self, dataset, num_samples):
            self.dataset = dataset  # public: dx_com parser traverses .dataset for transforms
            self._length = min(num_samples, len(dataset))

        def __len__(self):
            return self._length

        def __getitem__(self, idx):
            item = self.dataset[idx]
            inp = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp.astype(np.float32))
            elif not isinstance(inp, torch.Tensor):
                inp = torch.tensor(inp, dtype=torch.float32)
            else:
                inp = inp.float()
            if constant_inputs:
                return {primary_input_name: inp, **constant_inputs}
            return inp

    from torch.utils.data import DataLoader

    calib_ds = _CalibrationDataset(dataset, num_samples)
    return DataLoader(calib_ds, batch_size=1, shuffle=False, num_workers=0)
