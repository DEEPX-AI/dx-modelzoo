from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple

from dx_modelzoo.enums import SessionType
from dx_modelzoo.factory import ModelFactory
from dx_modelzoo.tools import run_benchmark as _run_benchmark
from dx_modelzoo.tools.benchmark_config import Modelzoo_config

cfg = Modelzoo_config()


def parse_session_type_and_path(args: Namespace) -> Tuple[SessionType, str, Optional[str]]:
    """parse runtime session type and path.
    it parses runtime session type and return it with file path.

    Args:
        args (Namespace): arguments.

    Raises:
        ValueError: if session type is not decided, raise ValueError.

    Returns:
        Tuple[SessionType, str, Optional[str]]: session type, file path, zero-shot text embedding path.
    """
    session_type = None
    path = None
    if args.onnx is not None:
        session_type = SessionType.onnxruntime
        path = args.onnx

    if args.dxnn is not None:
        session_type = SessionType.dxruntime
        path = args.dxnn
    if session_type is None:
        raise ValueError("Can't not Parse Session Type. check file path.")

    if args.zero_shot_text_embedding is not None:
        text_embedding_path = args.zero_shot_text_embedding
    else:
        text_embedding_path = None

    return session_type, path, text_embedding_path


def run_eval(args: Namespace) -> None:
    """run eval comman.

    Args:
        args (Namespace): arguments.
    """
    model_name = args.model_name
    session_type, model_path, text_embedding_path = parse_session_type_and_path(args)
    model = ModelFactory(model_name, session_type, model_path, args.data_dir, text_embedding_path).make_model()
    print(f"Run {model_name} Evaluation.\n")
    model.eval(debug_mode=args.debug)


def run_compile(args: Namespace) -> None:
    """run compile command.

    Args:
        args (Namespace): arguments.
    """

    try:
        from dx_com import compile
    except ImportError as e:
        if __debug__:
            raise e

        raise ImportError(
            " ❌ dx_com is not installed. "
            "    Please install dx_com to use the compile feature."
            "    You can install it via pip: pip install dx_com"
        )

    onnx_file = Path(args.onnx)
    if not (onnx_file.exists() and onnx_file.is_file()):
        raise FileNotFoundError(f" ❌ ONNX file not found: {onnx_file}")

    json_file = onnx_file.with_suffix(".json")
    if args.json is not None:
        json_file = Path(args.json)
    else:
        print(f" ⚠️ JSON config file not specified. Using default JSON file path: {json_file}")

    if not (json_file.exists() and json_file.is_file()):
        raise FileNotFoundError(f" ❌ JSON config file not found: {json_file}")

    output_dir = args.output
    if args.output is None:
        output_dir = "compiled"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compile(
        model=onnx_file.resolve().as_posix(),
        output_dir=output_dir.resolve().as_posix(),
        config=json_file.resolve().as_posix(),
        quantization_device="cuda" if args.use_gpu else "cpu",
        opt_level=1,
        gen_log=False,
    )


def run_info(args: Namespace) -> None:
    """run info command.

    Args:
        args (Namespace): arguments.

    Raises:
        ValueError: if model name is invalid, raise ValueError.
    """
    from dx_modelzoo.factory.dicts.model_dict import MODEL_DICT

    model_name = args.model_name
    if model_name in MODEL_DICT:
        model_cls = MODEL_DICT[model_name]
    else:
        raise ValueError(f"Invalid Model Name. {model_name}")
    model_cls.print_info()


def run_models(*args):
    """run models command."""
    from dx_modelzoo.factory.model_factory import MODEL_DICT

    print("Available Model List:")
    print(sorted(list(MODEL_DICT.keys())))


def run_benchmark(args: Namespace):
    print(args)
    _run_benchmark.main(args)


COMMAND_DICT = {
    "eval": run_eval,
    "info": run_info,
    "models": run_models,
    "benchmark": run_benchmark,
    "compile": run_compile,
}

__all__ = ["COMMAND_DICT"]
