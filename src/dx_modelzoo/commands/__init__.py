from argparse import Namespace
from typing import Tuple

from dx_modelzoo.enums import SessionType
from dx_modelzoo.factory import ModelFactory
from dx_modelzoo.tools import run_benchmark as _run_benchmark
from dx_modelzoo.tools.benchmark_config import Modelzoo_config

from dx_modelzoo.tools.benchmark_config import Modelzoo_config
cfg = Modelzoo_config()


def parse_session_type_and_path(args: Namespace) -> Tuple[SessionType, str]:
    """parse runtime session type and path.
    it parses runtime session type and return it with file path.

    Args:
        args (Namespace): arguments.

    Raises:
        ValueError: if session type is not decided, raise ValueError.

    Returns:
        Tuple[SessionType, str]: session type and file path.
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

    return session_type, path


def run_eval(args: Namespace) -> None:
    """run eval comman.

    Args:
        args (Namespace): arguments.
    """
    model_name = args.model_name
    session_type, model_path = parse_session_type_and_path(args)
    model = ModelFactory(model_name, session_type, model_path, args.data_dir).make_model()
    print(f"Run {model_name} Evaluation.\n")
    model.eval()


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

COMMAND_DICT = {"eval": run_eval, 
                "info": run_info, 
                "models": run_models,
                "benchmark": run_benchmark}

__all__ = ["COMMAND_DICT"]
