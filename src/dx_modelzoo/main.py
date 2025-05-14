from argparse import ArgumentParser, ArgumentTypeError
from dx_modelzoo.commands import COMMAND_DICT


def get_parser():
    parser = ArgumentParser(description="DEEPX ModelZoo arguments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval sub command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("model_name", type=str, help="Model name to run.")
    eval_parser.add_argument("--onnx", type=str, help="ONNX file path.")
    eval_parser.add_argument("--dxnn", type=str, help="DXNN file path. With this argument, you can run DXNN using the NPU.")
    eval_parser.add_argument("--data_dir", type=str, help="Dataset root dir.", required=True)

    # info sub command
    info_parser = subparsers.add_parser("info", help="Show model info")
    info_parser.add_argument("model_name", type=str, help="Model name to show info.")

    # models sub command
    subparsers.add_parser("models", help="Show available models")

    # benchmark sub command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    benchmark_parser.add_argument("model_name", nargs="?", type=str, help="Model name to benchmark.")
    benchmark_parser.add_argument("--data_dir", type=str, help="Dataset root dir.", required=True)
    benchmark_parser.add_argument('--dxrt', action='store_true', help='Run DXRT evaluation only')
    benchmark_parser.add_argument('--onnxrt', action='store_true', help='Run ONNXRT evaluation only')
    benchmark_parser.add_argument('--all', action='store_true', help='Run both DXRT and ONNXRT evaluations (default)')
    benchmark_parser.add_argument('--debug', action='store_true', help='Set debug mode (evaluate only one model per category)')

    return parser

def validate_args(args):
    # rule: At least one of --dxrt, --onnxrt, or --all must be specified for benchmark.
    if args.command == "benchmark":
        if not (args.dxrt or args.onnxrt or args.all):
            raise ArgumentTypeError("At least one of --dxrt, --onnxrt, or --all must be specified for benchmark.")
    if args.command == "eval":
        if not (bool(args.onnx) ^ bool(args.dxnn)):
            raise ArgumentTypeError("Exactly one of --onnx or --dxnn must be specified. Setting both or neither is not allowed.")


def main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)

    command = COMMAND_DICT[args.command]
    command(args)

