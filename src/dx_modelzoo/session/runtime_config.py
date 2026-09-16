from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Dict, Optional

from dx_modelzoo.common.exceptions import ConfigError

__all__ = ["RuntimeConfig", "OnnxRuntimeConfig", "DxnnRuntimeConfig"]


@dataclass
class RuntimeConfig:
    """Common runtime options shared by all session backends.

    Backend-specific options live on the subclasses (``OnnxRuntimeConfig`` /
    ``DxnnRuntimeConfig``).  ``RuntimeConfig.from_profile`` dispatches to the
    right subclass based on the profile's ``target``.

    ``batch_size`` is carried here because it is a ``runtime`` YAML field, but
    it is consumed by the evaluator (not the session).

    ``ASYNC_DEFAULT`` is the per-backend default used when the YAML omits the
    ``async`` option; subclasses override it (DXNN → True, ONNX → False).
    """

    ASYNC_DEFAULT: ClassVar[bool] = False

    device: Any = None
    batch_size: int = 1
    async_mode: Optional[bool] = None
    start_method: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.start_method, str) or self.start_method not in {"auto", "forkserver", "spawn"}:
            raise ConfigError(
                f"runtime.start_method must be one of: auto, forkserver, spawn; got {self.start_method!r}"
            )
        if self.start_method != "auto" and self.start_method not in mp.get_all_start_methods():
            raise ConfigError(
                f"Unsupported multiprocessing start method {self.start_method!r}. "
                f"Supported on this platform: {sorted(mp.get_all_start_methods())}"
            )

    @property
    def use_async(self) -> bool:
        """Resolved async flag: the explicit value, else the backend default."""
        return self.async_mode if self.async_mode is not None else self.ASYNC_DEFAULT

    @staticmethod
    def _common_kwargs(runtime: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "device": runtime.get("device"),
            "batch_size": runtime.get("batch_size", 1),
            "async_mode": runtime.get("async"),
            "start_method": runtime.get("start_method", "auto"),
        }

    @classmethod
    def from_profile(cls, profile: Optional[Dict[str, Any]], target: Optional[str] = None) -> "RuntimeConfig":
        """Build the backend-appropriate config from a YAML profile.

        Args:
            profile: A profile dict (reads its ``runtime`` block).
            target: Optional explicit target ("onnx"/"dxnn").  When omitted it
                is inferred from ``profile.target`` (defaulting to onnx).
        """
        profile = profile or {}
        runtime = profile.get("runtime") or {}
        target = target or profile.get("target")
        if target == "dxnn":
            return DxnnRuntimeConfig.from_runtime(runtime)
        return OnnxRuntimeConfig.from_runtime(runtime)

    @classmethod
    def from_runtime(cls, runtime: Optional[Dict[str, Any]]) -> "RuntimeConfig":
        kwargs = cls._common_kwargs(runtime or {})
        return cls(**kwargs)

    def resolved_start_method(self) -> str:
        """Resolve auto to best supported portable worker start method."""
        if self.start_method != "auto":
            if self.start_method not in mp.get_all_start_methods():
                raise ConfigError(
                    f"Unsupported multiprocessing start method {self.start_method!r}. "
                    f"Supported on this platform: {sorted(mp.get_all_start_methods())}"
                )
            return self.start_method
        methods = mp.get_all_start_methods()
        if "forkserver" in methods:
            return "forkserver"
        if "spawn" in methods:
            return "spawn"
        raise ConfigError("No supported multiprocessing start method found; expected forkserver or spawn")

    def with_device_override(self, device: Any) -> "RuntimeConfig":
        """Return a copy with ``device`` replaced, unless ``device`` is ``None``."""
        if device is None:
            return self
        return replace(self, device=device)


@dataclass
class OnnxRuntimeConfig(RuntimeConfig):
    """ONNX Runtime session options (no backend-specific fields yet)."""


@dataclass
class DxnnRuntimeConfig(RuntimeConfig):
    """DXNN (dx_engine) session options.

    Attributes:
        buffer_count: YAML ``buffer_count`` → engine I/O buffer count
            (``InferenceOption.buffer_count``).  ``None`` keeps the engine
            default (6).
        use_ort: YAML ``use_ort`` → run unsupported ops on ONNX Runtime
            (``InferenceOption.use_ort``).  Defaults to ``True``.
    """

    ASYNC_DEFAULT: ClassVar[bool] = True

    buffer_count: Optional[int] = None
    use_ort: bool = True

    @classmethod
    def from_runtime(cls, runtime: Optional[Dict[str, Any]]) -> "DxnnRuntimeConfig":
        runtime = runtime or {}
        return cls(
            **cls._common_kwargs(runtime),
            buffer_count=runtime.get("buffer_count"),
            use_ort=runtime.get("use_ort", True),
        )
