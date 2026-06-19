from __future__ import annotations

import json
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from tqdm import tqdm

from dx_modelzoo.loader.discovery import ModelEntry, discover_models
from dx_modelzoo.loader.model_builder import ModelBuilder


def _eval_single(
    entry: ModelEntry,
    profile_name: str,
    data_root: Optional[str],
    device: Optional[Union[int, str]] = None,
) -> Dict[str, Any]:
    """Evaluate a single model entry. Returns a result dict."""
    try:
        tag = f"{entry.domain}/{entry.task}/{entry.name}"
        builder = ModelBuilder(entry.yaml_path, resolve_env=True)

        if profile_name not in builder.config.get("profiles", {}):
            logger.warning("  SKIP: profile '{}' not found", profile_name)
            return {
                "model": entry.name,
                "domain": entry.domain,
                "task": entry.task,
                "status": "skipped",
                "reason": f"profile '{profile_name}' not found",
            }

        metrics = builder.run_eval(
            profile_name=profile_name,
            data_dir=data_root,
            device=device,
        )

        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        # start = metrics.get("start_time", "").replace(" ", "_").replace(":", "-")
        fname = result_dir / f"eval_{builder.name}_{profile_name}.json"
        with open(fname, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.success(f"Saved to {fname}")

        return {
            "model": entry.name,
            "domain": entry.domain,
            "task": entry.task,
            "device": device,
            "status": "success",
            "performance": metrics,
        }
    except Exception as e:
        logger.error("  ERROR ({}): {}", tag, e)
        return {
            "model": entry.name,
            "domain": entry.domain,
            "task": entry.task,
            "device": device,
            "status": "error",
            "error": str(e),
        }


def run_benchmark(
    models_dir: Path,
    profile_name: str,
    data_root: Optional[str] = None,
    model_root: Optional[str] = None,
    domain: Optional[str] = None,
    task: Optional[str] = None,
    devices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Run evaluation across multiple models and collect results.

    Args:
        models_dir: Path to models directory
        profile_name: Profile name to use for all models
        data_root: Override DATA_ROOT for dataset paths
        model_root: Override MODEL_ROOT for model paths
        domain: Filter by domain (e.g., "cv")
        task: Filter by task (e.g., "classification")
        devices: List of device IDs for parallel execution.
                 Each device runs one model at a time.
                 e.g. [0, 1, 2, 3] runs up to 4 models in parallel.
                 If None/empty, runs sequentially (original behavior).

    Returns:
        List of result dicts with model name, metrics, etc.
    """
    entries = discover_models(models_dir, domain=domain, task=task)
    if not entries:
        logger.warning("No models found matching filters.")
        return []

    if devices and len(devices) > 0:
        logger.info(
            "Parallel benchmark: {} models across {} devices {}",
            len(entries),
            len(devices),
            devices,
        )
        results = _run_parallel(entries, profile_name, data_root, devices)
    else:
        results = _run_sequential(entries, profile_name, data_root)

    _print_summary(results, profile_name)
    return results


def _run_sequential(
    entries: List[ModelEntry],
    profile_name: str,
    data_root: Optional[str],
) -> List[Dict[str, Any]]:
    """Run models one by one (original behavior)."""
    results: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(entries), desc="Benchmarking models sequentially")
    for i, entry in enumerate(entries):
        ######################################################################
        # NOTE: The following code is duplicated in _eval_single() for logging purposes.

        find_files = list(Path("result").rglob(f"eval_{entry.name}_{profile_name}.json"))
        if len(find_files) > 0:
            with open(find_files[0], "r") as f:
                data = json.load(f)
                cached_profile = data.get("profile", "")
                performance = data.get("metrics", [])
                fps = data.get("fps", -1)

            if cached_profile == Path(find_files[0]).stem.split("_")[-1] and performance != []:
                logger.info(f"  ⏩ SKIP: Found existing result for {entry.name}, loading metrics")
                results.append(
                    {
                        "model": entry.name,
                        "domain": entry.domain,
                        "task": entry.task,
                        "device": None,
                        "status": "success",
                        "performance": performance,
                        "fps": fps,
                    }
                )

                shutil.move(find_files[0], Path("result") / f"eval_{entry.name}_{profile_name}.json")

                pbar.update(1)
                continue

        ######################################################################
        tag = f"{entry.domain}/{entry.task}/{entry.name}"
        logger.info("=" * 60)
        logger.info(f"Benchmarking({i + 1}/{len(entries)}): {tag}")
        logger.info("=" * 60)

        result = _eval_single(entry, profile_name, data_root)
        results.append(result)
        pbar.update(1)
    pbar.close()
    return results


def _run_parallel(
    entries: List[ModelEntry],
    profile_name: str,
    data_root: Optional[str],
    devices: List[int],
) -> List[Dict[str, Any]]:
    """Run models in parallel — one model per device at a time.

    Uses a device pool (queue) so that when a device finishes one model,
    it picks up the next pending model automatically.
    """
    device_pool: queue.Queue[int] = queue.Queue()
    for d in devices:
        device_pool.put(d)

    def _worker(index: int, entry: ModelEntry) -> Dict[str, Any]:
        device = device_pool.get()
        try:
            tag = f"{entry.domain}/{entry.task}/{entry.name}"
            device_tag = f" [device {device}]" if device is not None else ""
            logger.info("=" * 60)
            logger.info(f"Benchmark({index + 1}/{len(entries)}): {tag}{device_tag}")
            logger.info("=" * 60)

            return _eval_single(entry, profile_name, data_root, device=device)
        finally:
            device_pool.put(device)

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        future_to_entry = {executor.submit(_worker, i, entry): (i, entry) for i, entry in enumerate(entries)}
        for future in as_completed(future_to_entry):
            results.append(future.result())

    # Sort by original order (domain/task/name) for consistent output
    entry_order = {(e.domain, e.task, e.name): i for i, e in enumerate(entries)}
    results.sort(key=lambda r: entry_order.get((r["domain"], r["task"], r["model"]), 0))

    return results


def _print_summary(results: List[Dict[str, Any]], profile_name: str) -> None:
    """Print benchmark summary table and save to JSON."""
    summary_rows: List[Dict[str, Any]] = []

    logger.info("=" * 100)
    logger.info("{:<30} {:<10} {:<50} {:<10}", "Model", "Status", "Metrics", "FPS")
    logger.info("-" * 100)
    for r in results:
        model = r["model"]
        status = r["status"]
        if status == "success":
            perf = r.get("performance", {})
            # performance can be either a list of metrics or a dict with 'metrics' key
            if isinstance(perf, dict):
                metrics_list = perf.get("metrics", [])
                fps = perf.get("fps", r.get("fps", -1))
            else:
                metrics_list = perf
                fps = r.get("fps", -1)
            metrics_str = (
                ", ".join(f"{v['name']}:{v['metric_value']:.2f}" for v in metrics_list) if metrics_list else "N/A"
            )
            metrics_str = metrics_str[:47] + "..." if len(metrics_str) > 50 else metrics_str
            logger.info("{:<30} {:<10} {:<50} {:<10.1f}", model, status, metrics_str, fps)
            summary_rows.append({"model": model, "status": status, "metrics": metrics_str, "fps": fps})
        else:
            reason = r.get("reason", r.get("error", ""))
            logger.info("{:<30} {:<10} {:<50}", model, status, reason)
            summary_rows.append({"model": model, "status": status, "reason": reason})
    logger.info("=" * 100)

    # Save summary to JSON
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = result_dir / f"benchmark_{profile_name}_summary_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    logger.success(f"Summary saved to {fname}")
