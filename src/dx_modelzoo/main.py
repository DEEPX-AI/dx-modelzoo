from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import typer
from dotenv import load_dotenv
from loguru import logger

from dx_modelzoo.common import get_builtin_models_dir, get_workspace_custom_dir
from dx_modelzoo.loader.discovery import discover_models, effective_models, resolve_model, scan_all_models
from dx_modelzoo.loader.model_builder import ModelBuilder

app = typer.Typer(
    name="dxmz",
    help="dx-modelzoo CLI",
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _has_workspace_custom_dir() -> bool:
    return get_workspace_custom_dir().exists()


def _apply_env_overrides(
    data_root: Optional[str] = None,
    model_root: Optional[str] = None,
) -> None:
    """Apply CLI args to env vars. CLI args always override .env / existing env."""
    if data_root:
        os.environ["DATA_ROOT"] = data_root
    if model_root:
        os.environ["MODEL_ROOT"] = model_root


def _discover_workspace_entries(
    domain: Optional[str] = None,
    task: Optional[str] = None,
) -> List:
    builtin_models_dir = get_builtin_models_dir()
    if not _has_workspace_custom_dir():
        return discover_models(builtin_models_dir, domain=domain, task=task)

    builtin_entries = scan_all_models(builtin_models_dir, source="builtin")

    custom_root = get_workspace_custom_dir()
    custom_entries = scan_all_models(custom_root, source="custom") if custom_root.exists() else []

    entries = builtin_entries + custom_entries
    if domain:
        entries = [entry for entry in entries if entry.domain == domain]
    if task:
        entries = [entry for entry in entries if entry.task == task]
    return entries


def _discover_effective_workspace_entries(
    domain: Optional[str] = None,
    task: Optional[str] = None,
) -> List:
    entries = effective_models(_discover_workspace_entries(domain=domain, task=task))
    return sorted(entries, key=lambda entry: (entry.domain, entry.task, entry.name.lower()))


def resolve_model_target(
    model_arg: str,
    models_dir: Optional[Path] = None,
) -> ModelBuilder:
    """Resolve model argument to a ModelBuilder."""
    if model_arg.endswith((".yaml", ".yml")) or "/" in model_arg:
        return ModelBuilder(model_arg)

    if models_dir is None:
        if not _has_workspace_custom_dir():
            models_dir = get_builtin_models_dir()
            entries = discover_models(models_dir, name=model_arg)
            if not entries:
                logger.error(f"Model '{model_arg}' not found. Use 'dxmz list' to see available models.")
                raise typer.Exit(code=1)
            if len(entries) > 1:
                logger.warning(f"Multiple models named '{model_arg}'. Using first.")

            return ModelBuilder(entries[0].yaml_path)

        try:
            entry = resolve_model(model_arg, _discover_workspace_entries())
        except ValueError as exc:
            logger.error(str(exc))
            raise typer.Exit(code=1) from exc

        if entry is None:
            logger.error(f"Model '{model_arg}' not found. Use 'dxmz list' to see available models.")
            raise typer.Exit(code=1)

        return ModelBuilder(entry.yaml_path)

    entries = discover_models(models_dir, name=model_arg)
    if not entries:
        logger.error(f"Model '{model_arg}' not found. Use 'dxmz list' to see available models.")
        raise typer.Exit(code=1)
    if len(entries) > 1:
        logger.warning(f"Multiple models named '{model_arg}'. Using first.")

    return ModelBuilder(entries[0].yaml_path)


def _parse_model_path(
    model_path_str: Optional[str],
) -> Optional[Union[str, Dict[str, str]]]:
    """Parse --model-path value.

    Returns:
        None if input is None.
        str if plain path (backward compatible).
        dict if 'stage=path' format, e.g. 'det=/path/det.dxnn,rec=/path/rec.dxnn'.
    """
    if model_path_str is None:
        return None
    if "=" not in model_path_str:
        return model_path_str
    result = {}
    for part in model_path_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise typer.BadParameter(f"Invalid stage=path format: '{part}'")
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            raise typer.BadParameter("Empty stage name in --model-path")
        if not val:
            raise typer.BadParameter(f"Empty path for stage '{key}' in --model-path")
        result[key] = val
    return result


@app.command()
def eval(
    models: List[str] = typer.Argument(..., help="Model name or YAML path"),
    profile: str = typer.Option(..., help="Profile name"),
    data_root: Optional[str] = typer.Option(None, "--data-root", help="Override DATA_ROOT"),
    model_root: Optional[str] = typer.Option(None, "--model-root", help="Override MODEL_ROOT"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="Path to model file (onnx/dxnn)"),
    save: bool = typer.Option(False, "--save", help="Save results to JSON file"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible evaluation"),
) -> None:
    """Evaluate a model on its dataset."""
    _apply_env_overrides(data_root, model_root)

    from dx_modelzoo.common.seed import set_seed

    set_seed(seed)

    if isinstance(models, str):
        models = [models]
    else:
        if model_path and len(models) > 1:
            logger.error(
                "Multiple models provided but --model-path is only one. "
                "Please specify one model or remove --model-path."
            )
            raise typer.Exit(code=1)

    parsed_model_path = _parse_model_path(model_path)

    for model in models:
        builder = resolve_model_target(model)

        # Validate profile/target match for single model path
        if isinstance(parsed_model_path, str):
            if builder.is_pipeline:
                profile_cfg = builder.get_profile(profile)
            else:
                profile_cfg = builder.config.get("profiles", {}).get(profile, {})
            if profile_cfg.get("target") != Path(parsed_model_path).suffix[1:]:
                logger.error(
                    f" ❌ Profile '{profile}' target '{profile_cfg.get('target')}' does not match "
                    f"model file extension '{Path(parsed_model_path).suffix[1:]}'"
                )
                raise typer.Exit(code=1)

        logger.info(f"Evaluating {builder.name} with profile {profile}...")
        result = builder.run_eval(
            profile_name=profile,
            model_path=parsed_model_path,
        )
        logger.info(f"Results: {result}")
        if save and result:
            result_dir = Path("result")
            result_dir.mkdir(exist_ok=True)
            fname = result_dir / f"eval_{builder.name}_{profile}.json"
            with open(fname, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved to {fname}")

        # Exit with code 1 if any model evaluation failed (error in result)
        if result.get("error", None) is not None:
            raise typer.Exit(code=1)


@app.command()
def compile(
    model: str = typer.Argument(..., help="Model name or YAML path"),
    profile: str = typer.Option(..., help="Profile name"),
    output: Optional[str] = typer.Option(None, "--output", help="Output directory"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="Path to ONNX model"),
    data_root: Optional[str] = typer.Option(None, "--data-root", help="Override DATA_ROOT"),
    model_root: Optional[str] = typer.Option(None, "--model-root", help="Override MODEL_ROOT"),
    use_gpu: bool = typer.Option(False, "--use-gpu", help="Use GPU for quantization"),
) -> None:
    """Compile a model for NPU deployment."""
    _apply_env_overrides(data_root, model_root)
    builder = resolve_model_target(model)

    profile_cfg = builder.config.get("profiles", {}).get(profile, {})
    target = profile_cfg.get("target", "onnx")
    if target == "onnx":
        logger.error(f" ❌ Profile '{profile}' targets ONNX — compilation is only supported for NPU (dxnn) profiles.")
        raise typer.Exit(code=1)

    from dx_modelzoo.command.compile import compile_model

    if builder.is_pipeline:
        parsed = _parse_model_path(model_path)
        model_path_dict = parsed if isinstance(parsed, dict) else {}
        for stage_name in builder.pipeline_stages:
            stage_mp = model_path_dict.get(stage_name)
            # Avoid output file collision: suffix stage name for .dxnn outputs
            stage_output = output
            if output and Path(output).suffix == ".dxnn":
                stem = Path(output).stem
                stage_output = str(Path(output).parent / f"{stem}_{stage_name}.dxnn")
            logger.info(f"Compiling {builder.name}/{stage_name} with profile {profile}...")
            compile_model(
                builder=builder,
                profile_name=profile,
                output=stage_output,
                model_path=stage_mp,
                use_gpu=use_gpu,
                stage=stage_name,
            )
    else:
        logger.info(f"Compiling {builder.name} with profile {profile}...")
        compile_model(
            builder=builder,
            profile_name=profile,
            output=output,
            model_path=model_path,
            use_gpu=use_gpu,
        )


@app.command()
def benchmark(
    profile: str = typer.Option(..., help="Profile name"),
    yaml_root: Optional[str] = typer.Option(None, "--models-dir", help="Directory containing model YAMLs"),
    data_root: Optional[str] = typer.Option(None, "--data-root", help="Data root directory"),
    model_root: Optional[str] = typer.Option(None, "--model-root", help="Model root directory"),
    domain: Optional[str] = typer.Option(None, help="Filter by domain"),
    task: Optional[str] = typer.Option(None, help="Filter by task"),
    devices: Optional[str] = typer.Option(
        None,
        "--devices",
        help="Comma-separated device IDs for parallel execution (e.g. '0,1,2,3'). "
        "Each device runs one model at a time.",
    ),
    save: bool = typer.Option(False, "--save", help="Save results to JSON file"),
) -> None:
    """Benchmark multiple models."""
    _apply_env_overrides(data_root, model_root)

    from dx_modelzoo.command.benchmark import run_benchmark

    if yaml_root is not None:
        models_dir = Path(yaml_root)
    else:
        models_dir = get_builtin_models_dir()

    # Parse devices string → list of ints
    device_list = None
    if devices:
        device_list = [int(d.strip()) for d in devices.split(",") if d.strip()]

    results = run_benchmark(
        models_dir=models_dir,
        profile_name=profile,
        data_root=data_root,
        model_root=model_root,
        domain=domain,
        task=task,
        devices=device_list,
    )

    # Save results
    if save and results:
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = result_dir / f"benchmark_{profile}_{current_time}.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.success(f"Saved to {fname}")


@app.command(name="create")
def create_yaml() -> None:
    """Create a custom model scaffold in ./custom."""
    try:
        from dx_modelzoo.command.create_yaml import create_yaml

        create_yaml()
    except typer.Exit:
        raise
    except Exception as exc:
        logger.error(f"Error during model creation: {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="list")
def list_models(
    domain: Optional[str] = typer.Option(None, help="Filter by domain"),
    task: Optional[str] = typer.Option(None, help="Filter by task"),
    all: bool = typer.Option(False, "--all", "-a", help="Plain text output (non-interactive)"),
) -> None:
    """List available models. Interactive tree view by default."""
    use_workspace_custom = _has_workspace_custom_dir()
    try:
        if use_workspace_custom or all or not sys.stdout.isatty():
            if use_workspace_custom:
                entries = _discover_effective_workspace_entries(domain=domain, task=task)
            else:
                entries = discover_models(get_builtin_models_dir(), domain=domain, task=task)
        else:
            entries = []
    except ValueError as exc:
        logger.error(str(exc))
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    if all or not sys.stdout.isatty():
        if not sys.stdout.isatty() and not all:
            logger.warning("Non-interactive terminal detected. Use --all for plain text output.")
        if not entries:
            logger.warning("No models found.")
            return
        for entry in entries:
            typer.echo(f"{entry.domain}/{entry.task}/{entry.name}")
        return

    if use_workspace_custom and not entries:
        logger.warning("No models found.")
        return

    if not use_workspace_custom:
        from dx_modelzoo.tui.model_browser import run_interactive_list

        result = run_interactive_list(get_builtin_models_dir(), domain=domain, task=task)
    else:
        from dx_modelzoo.tui.model_browser import ModelTreeBrowser

        result = ModelTreeBrowser(entries, domain=domain, task=task).run()
    if result and isinstance(result, dict):
        action = result["action"]
        model_name = result["model"]
        profile = result["profile"]
        logger.info(f"Running: dxmz {action} {model_name} --profile {profile}")
        if action == "eval":
            builder = resolve_model_target(model_name)
            result = builder.run_eval(profile_name=profile)
            logger.info(f"Results: {result}")
        elif action == "compile":
            from dx_modelzoo.command.compile import compile_model

            builder = resolve_model_target(model_name)
            compile_model(builder=builder, profile_name=profile)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model name or YAML path"),
) -> None:
    """Show model information."""
    builder = resolve_model_target(model)
    typer.echo(f"Model YAML: {builder.yaml_path}")
    typer.echo(f"Name: {builder.name}")
    typer.echo(f"Inputs: {builder.inputs}")
    typer.echo(f"Profiles: {list(builder.config['profiles'].keys())}")
    if "dataset" in builder.config:
        typer.echo(f"Dataset: {builder.config['dataset']}")


def main() -> None:
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
