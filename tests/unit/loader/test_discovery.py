"""Tests for dx_modelzoo.loader.discovery."""

import yaml

from dx_modelzoo.loader.discovery import ModelEntry, discover_models


class TestDiscoverModels:
    def _create_model(self, tmp_path, domain, task, family, name):
        model_dir = tmp_path / domain / task / family
        model_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = model_dir / f"{name}.yaml"
        yaml_path.write_text(yaml.dump({"name": name, "task": task}))
        return yaml_path

    def test_finds_models(self, tmp_path):
        self._create_model(tmp_path, "cv", "image_classification", "resnet", "resnet50_224x224")
        entries = discover_models(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "resnet50_224x224"
        assert entries[0].domain == "cv"
        assert entries[0].task == "image_classification"

    def test_filter_by_name(self, tmp_path):
        self._create_model(tmp_path, "cv", "image_classification", "resnet", "resnet50_224x224")
        self._create_model(tmp_path, "cv", "image_classification", "vit", "vit-b_224x224")
        entries = discover_models(tmp_path, name="vit-b_224x224")
        assert len(entries) == 1
        assert entries[0].name == "vit-b_224x224"

    def test_filter_by_task(self, tmp_path):
        self._create_model(tmp_path, "cv", "image_classification", "resnet", "resnet50")
        self._create_model(tmp_path, "cv", "object_detection", "yolo", "yolov8n")
        entries = discover_models(tmp_path, task="object_detection")
        assert len(entries) == 1
        assert entries[0].name == "yolov8n"

    def test_custom_ops_detected(self, tmp_path):
        yaml_path = self._create_model(tmp_path, "cv", "face_detection", "blazeface", "blazeface")
        (yaml_path.parent / "custom_ops.py").write_text("# custom ops")
        entries = discover_models(tmp_path)
        assert entries[0].custom_ops_path is not None

    def test_shallow_yaml_skipped(self, tmp_path):
        # YAML at depth < 4 should be skipped
        (tmp_path / "shallow.yaml").write_text(yaml.dump({"name": "x"}))
        entries = discover_models(tmp_path)
        assert len(entries) == 0

    def test_invalid_yaml_skipped(self, tmp_path):
        model_dir = tmp_path / "cv" / "cls" / "bad"
        model_dir.mkdir(parents=True)
        (model_dir / "broken.yaml").write_text("{{invalid yaml")
        entries = discover_models(tmp_path)
        assert len(entries) == 0
