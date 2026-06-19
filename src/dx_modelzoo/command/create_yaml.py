"""Create command implementation for dxmz CLI.

This module implements the `dxmz create` command, which creates custom model scaffolds
using an interactive wizard interface.

The command:
- Enforces interactive terminal requirement
- Runs the create wizard from dx_modelzoo.tui.create_wizard
- Honors cancellation cleanly
- Builds scaffold output from wizard result
- Writes YAML to ./custom/<domain>/<task>/<family-or-model>/<model>.yaml
- Preserves duplicate custom-name protection and overwrite confirmation
- Notifies when custom model shadows a builtin model name
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import typer
from loguru import logger

from dx_modelzoo.common import get_builtin_models_dir, get_workspace_custom_dir
from dx_modelzoo.loader.discovery import discover_models
from dx_modelzoo.loader.model_scaffold import (
    ClassificationScaffold,
    DuplicateCustomModelError,
    InvalidIdentifierError,
    ensure_no_duplicate_custom_name,
    write_classification_scaffold,
)
from dx_modelzoo.tui.create_wizard import run_create_wizard

if TYPE_CHECKING:
    from dx_modelzoo.tui.create_wizard import CreateWizardResult


def _can_prompt_interactively() -> bool:
    """Check if we're in an interactive terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _check_builtin_shadow(model_name: str) -> None:
    """Check if custom model will shadow a builtin model and notify user.

    This is an informational check only. If builtin model discovery fails
    due to filesystem issues, we silently skip the check rather than crash.
    """
    try:
        builtin_matches = discover_models(get_builtin_models_dir(), name=model_name)
        if builtin_matches:
            typer.echo(f"Note: custom model '{model_name}' will shadow builtin model at {builtin_matches[0].yaml_path}")
    except Exception as exc:
        # Shadow check is informational only; silently skip on failure
        logger.debug("Shadow check skipped due to discovery error: {}", exc)


def _wizard_result_to_scaffold(result: CreateWizardResult) -> ClassificationScaffold:
    """Convert wizard result to ClassificationScaffold."""
    return ClassificationScaffold(
        domain=result.domain,
        task=result.task,
        dataset_name=result.dataset_name,
        family=result.family,
        model_name=result.model_name,
        reference=result.reference,
        description=result.description,
        input_name=result.input_name,
        input_shape=result.input_shape,
        preprocessing_steps=result.preprocessing_steps,
        postprocessing_steps=result.postprocessing_steps,
        dataset_eval_path=result.dataset_eval_path,
        artifact_base_path=result.artifact_base_path,
        profile_choice=result.profile_choice,
    )


def create_yaml() -> None:
    """Create a custom model scaffold in ./custom.

    This function implements the core logic of the create command:
    1. Enforces interactive terminal requirement
    2. Runs the wizard
    3. Handles cancellation
    4. Checks for builtin model shadowing
    5. Validates identifiers and builds target path
    6. Checks for duplicate custom names (before overwrite prompt)
    7. Handles overwrite confirmation
    8. Writes the scaffold YAML

    Raises:
        typer.Exit: On any error or cancellation
    """
    # Enforce interactive terminal
    if not _can_prompt_interactively():
        typer.echo("dxmz create requires an interactive terminal.")
        raise typer.Exit(code=1)

    # Run the wizard
    wizard_result = run_create_wizard()

    # Handle cancellation
    if wizard_result is None:
        typer.echo("Create cancelled.")
        raise typer.Exit(code=0)

    # Check for builtin shadow
    _check_builtin_shadow(wizard_result.model_name)

    # Convert wizard result to scaffold
    spec = _wizard_result_to_scaffold(wizard_result)

    # Get custom root
    custom_root = get_workspace_custom_dir()

    # Build target path and validate identifiers
    from dx_modelzoo.loader.model_scaffold import build_custom_model_path

    try:
        target_path = build_custom_model_path(
            custom_root,
            spec.family,
            spec.model_name,
            domain=spec.domain,
            task=spec.task,
        )
    except (InvalidIdentifierError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    # Check for duplicate custom names before overwrite prompt
    try:
        ensure_no_duplicate_custom_name(custom_root, spec.model_name, target_path)
    except DuplicateCustomModelError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    # Handle overwrite confirmation
    overwrite = False
    if target_path.exists():
        overwrite = typer.confirm(f"{target_path} already exists. Overwrite?", default=False)
        if not overwrite:
            typer.echo("Aborted: overwrite declined.")
            raise typer.Exit(code=1)

    custom_ops_path = target_path.parent / "custom_ops.py"
    custom_ops_preexisting = custom_ops_path.exists()

    # Write the scaffold
    try:
        created_path = write_classification_scaffold(custom_root, spec, overwrite=overwrite)
    except (InvalidIdentifierError, DuplicateCustomModelError, FileExistsError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    typer.secho(f"✓ Created custom model scaffold: {created_path}", fg=typer.colors.GREEN)
    if not custom_ops_preexisting and custom_ops_path.exists():
        typer.secho(f"✓ Created custom ops helper: {custom_ops_path}", fg=typer.colors.GREEN)
    typer.secho(
        "⚠ Review and edit the generated YAML before use. It is a convenience scaffold and may need manual fixes.",
        fg=typer.colors.YELLOW,
    )


def register_create_command(app: typer.Typer) -> None:
    """Register the create command with the Typer app."""
    app.command(name="create")(create_yaml)
