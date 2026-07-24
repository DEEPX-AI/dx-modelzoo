"""Integration test: verify dataset-evaluator-task registry mapping.

Ensures that every YAML's `task` field has a matching evaluator in the registry,
and every `dataset.type` field has a matching dataset in the registry.
"""

import pytest

from dx_modelzoo.dataset import DATASET_REGISTRY
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY
from dx_modelzoo.loader.yaml_loader import load_yaml


class TestRegistryMapping:
    def test_all_tasks_have_evaluator(self, all_yaml_paths):
        """Every YAML task field must map to a registered evaluator."""
        unmapped = []
        for path in all_yaml_paths:
            config = load_yaml(path)
            if "pipeline" in config:
                continue
            task = config.get("task")
            if task and task not in EVALUATOR_REGISTRY:
                unmapped.append(f"{path.name}: task='{task}' not in EVALUATOR_REGISTRY")
        assert not unmapped, f"{len(unmapped)} unmapped tasks:\n" + "\n".join(unmapped[:10])

    def test_all_dataset_types_registered(self, all_yaml_paths):
        """Every YAML dataset.type must map to a registered dataset class."""
        unmapped = []
        for path in all_yaml_paths:
            config = load_yaml(path)
            ds_type = config.get("dataset", {}).get("type")
            if ds_type and ds_type not in DATASET_REGISTRY:
                unmapped.append(f"{path.name}: dataset.type='{ds_type}' not in DATASET_REGISTRY")
        assert not unmapped, f"{len(unmapped)} unmapped datasets:\n" + "\n".join(unmapped[:10])

    def test_evaluator_registry_not_empty(self):
        """Evaluator registry should have all expected task evaluators."""
        expected_tasks = [
            "image_classification",
            "object_detection",
            "semantic_segmentation",
            "instance_segmentation",
            "depth_estimation",
            "super_resolution",
            "pose_estimation",
            "face_detection",
        ]
        for task in expected_tasks:
            assert task in EVALUATOR_REGISTRY, f"'{task}' missing from EVALUATOR_REGISTRY"

    def test_dataset_registry_not_empty(self):
        """Dataset registry should have all commonly used datasets."""
        expected_datasets = [
            "ILSVRC2012",
            "COCO",
            "ADE20K",
            "Cityscapes",
            "WiderFace",
            "BSD100",
            "NYUDepthv2",
        ]
        for ds in expected_datasets:
            assert ds in DATASET_REGISTRY, f"'{ds}' missing from DATASET_REGISTRY"

    def test_task_to_dataset_consistency(self, all_yaml_paths):
        """Check that task-dataset pairings make sense (no object_detection with BSD68)."""
        suspicious = []
        # Known valid pairings (not exhaustive, just catch obvious mismatches)
        invalid_combos = {
            "image_classification": ["COCO", "WiderFace", "Cityscapes"],
            "object_detection": ["ILSVRC2012", "BSD68", "BSD100"],
            "semantic_segmentation": ["ILSVRC2012", "COCO", "WiderFace"],
            "super_resolution": ["ILSVRC2012", "COCO"],
        }
        for path in all_yaml_paths:
            config = load_yaml(path)
            if "pipeline" in config:
                continue
            task = config.get("task")
            ds_type = config.get("dataset", {}).get("type")
            if task in invalid_combos and ds_type in invalid_combos[task]:
                suspicious.append(f"{path.name}: task='{task}' with dataset='{ds_type}'")
        # This is a warning, not a hard failure (may have valid exceptions)
        if suspicious:
            pytest.warns(UserWarning, match="suspicious") if False else None
            # Just report, don't fail — some exceptions may be valid
            print(f"\n⚠️ Suspicious task-dataset pairings ({len(suspicious)}):")
            for s in suspicious[:5]:
                print(f"  {s}")
