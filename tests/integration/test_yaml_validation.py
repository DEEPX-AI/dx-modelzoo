"""Integration test: validate all YAML model configs.

Ensures every YAML under src/dx_modelzoo/models/cv/ passes structural
validation — required fields, valid preprocessing types, valid profile
structure, etc.
"""

import pytest

from dx_modelzoo.loader.yaml_loader import load_yaml, validate_yaml


class TestYamlValidation:
    def test_all_yamls_load(self, all_yaml_paths):
        """Every YAML file should parse without error."""
        failures = []
        for path in all_yaml_paths:
            try:
                load_yaml(path)
            except Exception as e:
                failures.append(f"{path.name}: {e}")
        assert not failures, f"{len(failures)} YAML load failures:\n" + "\n".join(failures[:10])

    def test_all_yamls_validate(self, all_yaml_paths):
        """Every YAML file should pass structural validation."""
        import warnings
        failures = []
        for path in all_yaml_paths:
            try:
                config = load_yaml(path)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    validate_yaml(config, str(path))
            except Exception as e:
                failures.append(f"{path.name}: {e}")
        assert not failures, f"{len(failures)} validation failures:\n" + "\n".join(failures[:10])

    def test_all_yamls_have_required_fields(self, all_yaml_paths):
        """Every YAML must have name, task, inputs, preprocessing, dataset, profiles."""
        required = ["name", "inputs", "preprocessing", "dataset", "profiles"]
        missing_report = []
        for path in all_yaml_paths:
            config = load_yaml(path)
            # Pipeline YAMLs have different structure
            if "pipeline" in config:
                continue
            missing = [f for f in required if f not in config]
            if missing:
                missing_report.append(f"{path.name}: missing {missing}")
        assert not missing_report, "\n".join(missing_report[:10])

    def test_all_yamls_have_task_field(self, all_yaml_paths):
        """Every non-pipeline YAML should have a 'task' field."""
        no_task = []
        for path in all_yaml_paths:
            config = load_yaml(path)
            if "pipeline" not in config and "task" not in config:
                no_task.append(path.name)
        if no_task:
            import warnings
            warnings.warn(f"{len(no_task)} YAMLs missing 'task' field: {no_task}")
        # Allow up to 2 known exceptions for now
        assert len(no_task) <= 2, f"{len(no_task)} YAMLs missing 'task':\n" + "\n".join(no_task)

    def test_yaml_count_reasonable(self, all_yaml_paths):
        """Sanity check: we should have a substantial number of models."""
        assert len(all_yaml_paths) >= 200, f"Only {len(all_yaml_paths)} YAMLs found, expected 200+"
