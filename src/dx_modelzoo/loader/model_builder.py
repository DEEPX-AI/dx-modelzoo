from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from dx_modelzoo.loader.config import resolve_variables_recursive

if TYPE_CHECKING:
    from dx_modelzoo.common.dataloader import DatasetBase
    from dx_modelzoo.session import SessionBase

from dx_modelzoo.loader.yaml_loader import load_yaml, validate_yaml
from dx_modelzoo.postprocessing import PostprocessingPipeline
from dx_modelzoo.preprocessing import PreprocessingPipeline


class ModelBuilder:
    """Builds runnable model components from a YAML config file."""

    def __init__(self, yaml_path: Union[str, Path], resolve_env: bool = True) -> None:
        self.yaml_path = Path(yaml_path)
        self.model_dir = self.yaml_path.parent

        self.config = load_yaml(self.yaml_path)
        if resolve_env:
            self.config = resolve_variables_recursive(self.config, str(self.yaml_path))

        # Auto-import custom_ops.py before validation (so custom types are registered)
        self._import_custom_ops()
        validate_yaml(self.config, str(self.yaml_path))

    def _import_custom_ops(self) -> None:
        """Auto-import custom_ops.py from model directory if it exists."""
        custom_ops_path = self.model_dir / "custom_ops.py"
        if custom_ops_path.exists():
            resolved_custom_ops = custom_ops_path.resolve()
            module_key = hashlib.sha1(str(custom_ops_path.resolve()).encode("utf-8")).hexdigest()[:12]
            module_name = f"custom_ops_{self.model_dir.name}_{module_key}"
            if module_name in sys.modules:
                return
            for loaded_name, loaded_module in sys.modules.items():
                loaded_file = getattr(loaded_module, "__file__", None)
                if loaded_file is None:
                    continue
                try:
                    if Path(loaded_file).resolve() == resolved_custom_ops:
                        sys.modules[module_name] = loaded_module
                        return
                except OSError:
                    continue
            spec = importlib.util.spec_from_file_location(module_name, custom_ops_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load custom_ops from {custom_ops_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def inputs(self) -> List[Dict[str, Any]]:
        if self.is_pipeline:
            first_stage = list(self.config["pipeline"].keys())[0]
            return self.config["pipeline"][first_stage]["inputs"]
        return self.config["inputs"]

    @property
    def appendix(self) -> Dict[str, Any]:
        """Model-specific overrides (e.g. npu_skip_arithmetic, label_preprocessing)."""
        val = self.config.get("appendix", {})
        return val if isinstance(val, dict) else {}

    @property
    def is_pipeline(self) -> bool:
        """True if this YAML defines a multi-model pipeline."""
        return "pipeline" in self.config

    @property
    def pipeline_stages(self) -> List[str]:
        """Ordered list of pipeline stage names."""
        if not self.is_pipeline:
            return []
        return list(self.config["pipeline"].keys())

    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """Get the config dict for a specific pipeline stage."""
        return self.config["pipeline"][stage]

    def get_profile(self, profile_name: str, stage: Optional[str] = None) -> Dict[str, Any]:
        profiles = self.config["profiles"]
        if profile_name not in profiles:
            raise KeyError(f"Profile '{profile_name}' not found. " f"Available: {list(profiles.keys())}")
        return profiles[profile_name]

    def build_preprocessing(
        self,
        profile_name: str,
        modal: Optional[str] = None,
        input_dtype: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> PreprocessingPipeline:
        """Build preprocessing pipeline for given profile.

        Args:
            input_dtype: Runtime input dtype (e.g. 'uint8', 'float32').
                For dxnn targets, uint8 means NPU absorbed arithmetic ops → skip them.
                float32 means NPU expects pre-normalized input → keep them.
        """
        profile = self.get_profile(profile_name, stage=stage)
        target = profile.get("target", "")
        npu_mode = target == "dxnn"

        # Determine skip behavior from dtype when running on NPU
        # uint8 → NPU handles normalization internally → skip arithmetic
        # float32 → NPU expects normalized input → keep arithmetic
        #
        # Some models (e.g. ViT, DeiT, ResMLP) need normalization even with
        # uint8 NPU input.  Override via ``appendix.npu_skip_arithmetic: false``.
        yaml_override = self.appendix.get("npu_skip_arithmetic")
        if yaml_override is not None:
            npu_skip_arithmetic = bool(yaml_override) and npu_mode
        elif npu_mode and input_dtype is not None:
            import numpy as np

            npu_skip_arithmetic = np.dtype(input_dtype) == np.dtype("uint8")
        else:
            npu_skip_arithmetic = npu_mode  # fallback: skip by default for NPU

        if stage and self.is_pipeline:
            preproc_config = self.config["pipeline"][stage]["preprocessing"]
        else:
            preproc_config = self.config["preprocessing"]

        # Handle multimodal preprocessing — select single modal or pass dict through
        if isinstance(preproc_config, dict) and modal:
            preproc_config = preproc_config.get(modal, [])

        if npu_skip_arithmetic:
            preproc_config = self._apply_npu_substitutions(preproc_config)

        return PreprocessingPipeline(preproc_config, npu_mode=npu_skip_arithmetic)

    @staticmethod
    def _apply_npu_substitutions(config):
        """Replace ops for NPU mode (e.g. bgr_to_y_channel → uint8 variant).

        Handles both list (single-input) and dict (multi-input) configs.
        """

        def _sub_steps(steps):
            return [
                {**step, "type": "bgr_to_y_channel_uint8"} if step["type"] == "bgr_to_y_channel" else dict(step)
                for step in steps
            ]

        if isinstance(config, list):
            return _sub_steps(config)
        if isinstance(config, dict):
            return {name: _sub_steps(steps) if isinstance(steps, list) else steps for name, steps in config.items()}
        return config

    def build_postprocessing(self, stage: Optional[str] = None) -> PostprocessingPipeline:
        """Build postprocessing pipeline."""
        if stage and self.is_pipeline:
            config = self.config["pipeline"][stage].get("postprocessing", [])
        else:
            config = self.config.get("postprocessing", [])
        if config is None:
            config = []
        return PostprocessingPipeline(config)

    def build_session(
        self,
        profile_name: str,
        model_path: Optional[str] = None,
        device: Optional[Union[int, str, list]] = None,
        stage: Optional[str] = None,
    ) -> "SessionBase":
        """Build session from profile or direct model path.

        If model_path is given, creates session directly.
        Otherwise, delegates to session factory for profile-based resolution
        (including auto-download if DXMZ_MODEL_URL is set).

        Args:
            profile_name: Profile name in YAML config.
            model_path: Direct path to model file (.onnx/.dxnn).
            device: Override device (e.g. 0, "0,1", [0, 1]).
                    If None, uses profile's runtime.device.
            stage: Pipeline stage name (for pipeline YAMLs).
        """
        from dx_modelzoo.session.factory import create_session
        from dx_modelzoo.session.runtime_config import RuntimeConfig

        if stage and self.is_pipeline:
            return self._build_pipeline_stage_session(profile_name, stage, device)

        profile = self.get_profile(profile_name, stage=stage)
        runtime_config = RuntimeConfig.from_profile(profile).with_device_override(device)

        if model_path is not None:
            return create_session(model_path, runtime_config=runtime_config)

        return create_session(profile_name, builder=self, runtime_config=runtime_config)

    def _build_pipeline_stage_session(
        self,
        profile_name: str,
        stage: str,
        device: Optional[Union[int, str, list]] = None,
    ) -> "SessionBase":
        """Build a session for a specific pipeline stage using its own artifacts/profiles."""
        from loguru import logger

        from dx_modelzoo.session.factory import SessionCreationError, _create_session, _try_download
        from dx_modelzoo.session.runtime_config import RuntimeConfig

        stage_cfg = self.get_stage_config(stage)
        profile = self.get_profile(profile_name)
        target = profile.get("target")
        if target is None:
            target = "onnx" if profile_name == "onnx" else "dxnn"

        runtime_config = RuntimeConfig.from_profile(profile, target=target).with_device_override(device)

        artifacts = stage_cfg.get("artifacts", {})
        base_path_str = artifacts.get("path") or artifacts.get(target)
        if not base_path_str:
            raise SessionCreationError(f"No artifact path for pipeline stage '{stage}' in {self.yaml_path}")
        base_path = Path(base_path_str)

        if target == "onnx":
            model_path = base_path if base_path.suffix == ".onnx" else base_path / f"{base_path.stem}.onnx"
        else:
            # DXNN: {stem}_{profile}.dxnn in same directory as the ONNX artifact
            stem = base_path.stem  # e.g. PP_OCRv5_Det_640
            dxnn_dir = base_path.parent if base_path.suffix else base_path
            model_path = dxnn_dir / f"{stem}_{profile_name}.dxnn"

        if model_path.exists():
            logger.info(f"Creating {target} session for pipeline stage '{stage}': {model_path}")
            return _create_session(str(model_path), target, runtime_config)

        logger.warning(f"Pipeline stage '{stage}' model not found: {model_path}")
        _try_download(base_path.stem, profile_name, model_path)
        return _create_session(str(model_path), target, runtime_config)

    def build_dataset(self, data_dir: Optional[str] = None) -> "DatasetBase":
        """Build dataset from YAML config."""
        import inspect

        from dx_modelzoo.dataset import DATASET_REGISTRY

        ds_config = self.config.get("dataset", {})
        ds_type = ds_config.get("type")
        if ds_type is None:
            raise ValueError(f"No dataset type specified in {self.yaml_path}")

        ds_cls = DATASET_REGISTRY.get(ds_type)
        ds_path = data_dir or ds_config.get("eval_path", "")

        # Forward any extra dataset.* fields (other than type/eval_path)
        # as kwargs if the dataset's __init__ accepts them.  Inject the
        # model-level ``inputs`` spec so synthetic / multi-input
        # datasets can build the right tensors.
        extra_kwargs = {k: v for k, v in ds_config.items() if k not in ("type", "eval_path")}
        try:
            sig = inspect.signature(ds_cls.__init__)
            params = sig.parameters
            if "inputs" in params and "inputs" not in extra_kwargs:
                extra_kwargs["inputs"] = self.config.get("inputs", [])
            extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in params}
        except (TypeError, ValueError):
            extra_kwargs = {}
        return ds_cls(ds_path, **extra_kwargs)

    def build_evaluator(
        self,
        session: "SessionBase",
        dataset: "DatasetBase",
        profile_name: Optional[str] = None,
    ):
        """Build evaluator based on model's task (inferred from directory path)."""
        from dx_modelzoo.evaluator import EVALUATOR_REGISTRY

        parts = self.yaml_path.parts
        custom_root_index = self._custom_root_index(parts)

        if custom_root_index is not None:
            eval_type = self.config.get("evaluator", {}).get("type")
            if not eval_type:
                raise ValueError(f"Custom model YAMLs under ./custom must define evaluator.type: {self.yaml_path}")
        else:
            # Priority 1: explicit evaluator.type override
            eval_type = self.config.get("evaluator", {}).get("type")

            # Priority 2: YAML 'task' field
            if not eval_type:
                eval_type = self.config.get("task")

            # Priority 3: Infer from YAML file's parent.parent directory name
            if not eval_type:
                eval_type = (
                    self.yaml_path.parent.parent.name if self.yaml_path.parent.parent != self.yaml_path.parent else None
                )

            if not eval_type:
                raise ValueError(
                    f"Cannot determine evaluator type for {self.yaml_path}. "
                    "Please specify task: <task_name> or evaluator.type: <type_name> in the YAML config."
                )

        evaluator_cls = EVALUATOR_REGISTRY.get(eval_type)

        # Read batch_size from profile runtime config
        # dxnn target only supports batch_size=1
        batch_size = 1
        if profile_name:
            from dx_modelzoo.session.runtime_config import RuntimeConfig

            profile = self.get_profile(profile_name)
            if profile.get("target") != "dxnn":
                batch_size = RuntimeConfig.from_profile(profile).batch_size

        evaluator = evaluator_cls(session, dataset, batch_size=batch_size)

        # Set model spec for evaluator (macs, params, input_resolution, license)
        input_shape = self.config.get("inputs")[0].get("shape")
        if len(input_shape) == 4:
            input_resolution = tuple(input_shape[2:] + input_shape[1:2])  # H, W, C
        elif len(input_shape) < 4:
            input_resolution = tuple(input_shape[1:])  # H, W / C
        else:
            input_resolution = tuple(input_shape)  # Fallback for unexpected shapes
        evaluator.model_spec = {
            "operations": self.config.get("macs"),
            "parameters": self.config.get("params"),
            "input_resolution": input_resolution,
            "license": self.config.get("license", None),
        }

        # Apply appendix properties to evaluator (by attribute name)
        for key, value in self.appendix.items():
            if hasattr(type(evaluator), key) or hasattr(evaluator, key):
                setattr(evaluator, key, value)

        # Apply evaluator.options to evaluator
        # Skip NPU-only options (use_ppu) for non-NPU profiles
        is_npu = profile_name and self.get_profile(profile_name).get("target") == "dxnn"
        eval_options = self.config.get("evaluator", {}).get("options", {})
        if isinstance(eval_options, dict):
            npu_only_keys = {"use_ppu", "anchors", "yolo_version"}
            for key, value in eval_options.items():
                if key in npu_only_keys and not is_npu:
                    continue
                setattr(evaluator, key, value)

        # Resolve relative file paths in evaluator options against YAML directory
        for key in ("dict_path",):
            val = getattr(evaluator, key, "")
            if val and not os.path.isabs(val) and not os.path.isfile(val):
                resolved = self.model_dir / val
                if resolved.is_file():
                    setattr(evaluator, key, str(resolved))

        # Set is_npu flag based on profile target
        if profile_name:
            profile = self.get_profile(profile_name)
            if hasattr(evaluator, "__dict__"):
                evaluator.is_npu = profile.get("target") == "dxnn"

        return evaluator

    @staticmethod
    def _custom_root_index(parts: tuple[str, ...]) -> Optional[int]:
        """Find the index of 'custom' in path that forms custom/<domain>/<task>/<family>/<file>.yaml.

        Returns the index of 'custom' if followed by exactly 4 more parts (domain/task/family/file),
        regardless of whether the domain exists in builtin models. Excludes paths that contain
        'models' directory after 'custom' (those follow builtin structure).
        """
        for i, part in enumerate(parts):
            if part != "custom":
                continue
            tail = parts[i:]
            # custom/<domain>/<task>/<family>/<file>.yaml = 5 parts total
            # Exclude if path contains 'models' after 'custom' (e.g., custom/workspace/models/...)
            if len(tail) == 5 and "models" not in tail[1:]:
                return i
        return None

    def run_eval(
        self,
        profile_name: str,
        data_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[Union[int, str, list]] = None,
    ) -> dict:
        """Full evaluation pipeline: session -> dataset -> evaluator -> run.

        Args:
            profile_name: Profile name in YAML config.
            data_dir: Override dataset directory.
            model_path: Direct path to model file.
            device: Override device for session creation.
        """
        if self.is_pipeline:
            return self._run_pipeline_eval(profile_name, data_dir, model_path, device)

        session = self.build_session(profile_name, model_path, device=device)
        dataset = self.build_dataset(data_dir)

        # Apply label_preprocessing from appendix to dataset if present
        label_preproc = self.appendix.get("label_preprocessing")
        if label_preproc and hasattr(type(dataset), "label_preprocessing"):
            dataset.label_preprocessing = PreprocessingPipeline(label_preproc, npu_mode=False)
        hr_preproc = self.appendix.get("hr_preprocessing")
        if hr_preproc and hasattr(dataset, "set_hr_preprocessing"):
            dataset.set_hr_preprocessing(PreprocessingPipeline(hr_preproc, npu_mode=False))

        evaluator = self.build_evaluator(session, dataset, profile_name)

        # Set evaluation context
        evaluator.model_name = self.name
        evaluator.display_name = self.config.get("display_name")
        evaluator.dataset_name = self.config.get("dataset", {}).get("type", "")
        evaluator.profile_name = profile_name

        # Get input dtype from session for preprocessing decisions
        input_dtype = None
        if hasattr(session, "dtype") and session.dtype:
            input_dtype = session.dtype[0]

        preprocessing = self.build_preprocessing(profile_name, input_dtype=input_dtype)
        evaluator.set_preprocessing(preprocessing)
        evaluator.set_postprocessing(self.build_postprocessing())

        return evaluator.eval()

    def _run_pipeline_eval(
        self,
        profile_name: str,
        data_dir: Optional[str] = None,
        model_path=None,
        device=None,
    ) -> dict:
        """Run evaluation for a pipeline YAML (multiple stages)."""
        # model_path: dict {stage: path} or str (applied to first stage) or None
        model_path_dict: Dict[str, Optional[str]] = {}
        if isinstance(model_path, dict):
            model_path_dict = model_path
        elif isinstance(model_path, str):
            model_path_dict[self.pipeline_stages[0]] = model_path

        # Build session + preprocessing for each stage
        pipeline_sessions: Dict[str, "SessionBase"] = {}
        pipeline_preprocessings = {}

        for stage_name in self.pipeline_stages:
            stage_mp = model_path_dict.get(stage_name)
            session = self.build_session(profile_name, model_path=stage_mp, device=device, stage=stage_name)
            pipeline_sessions[stage_name] = session

            input_dtype = None
            if hasattr(session, "dtype") and session.dtype:
                input_dtype = session.dtype[0]
            pipeline_preprocessings[stage_name] = self.build_preprocessing(
                profile_name, input_dtype=input_dtype, stage=stage_name
            )

        # Primary stage = first stage
        primary_stage = self.pipeline_stages[0]
        primary_session = pipeline_sessions[primary_stage]

        dataset = self.build_dataset(data_dir)

        # Apply label_preprocessing from appendix
        label_preproc = self.appendix.get("label_preprocessing")
        if label_preproc and hasattr(type(dataset), "label_preprocessing"):
            dataset.label_preprocessing = PreprocessingPipeline(label_preproc, npu_mode=False)

        evaluator = self.build_evaluator(primary_session, dataset, profile_name)

        # Inject pipeline context into evaluator
        evaluator.pipeline_sessions = pipeline_sessions
        evaluator.pipeline_preprocessings = pipeline_preprocessings

        evaluator.model_name = self.name
        evaluator.display_name = self.config.get("display_name")
        evaluator.dataset_name = self.config.get("dataset", {}).get("type", "")
        evaluator.profile_name = profile_name

        evaluator.set_preprocessing(pipeline_preprocessings[primary_stage])
        evaluator.set_postprocessing(self.build_postprocessing(stage=primary_stage))

        return evaluator.eval()
