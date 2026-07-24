"""Tests for dx_modelzoo.command.compile."""

import numpy as np

from dx_modelzoo.command.compile import _remove_expanddim


class TestRemoveExpanddim:
    def test_removes_expanddim_from_list(self):
        config = [
            {"type": "resize", "size": [224, 224]},
            {"type": "expanddim", "axis": 0},
            {"type": "div", "x": 255},
        ]
        result = _remove_expanddim(config)
        types = [s["type"] for s in result]
        assert "expanddim" not in types
        assert "resize" in types
        assert "div" in types

    def test_no_expanddim(self):
        config = [{"type": "resize", "size": [224, 224]}]
        result = _remove_expanddim(config)
        assert len(result) == 1

    def test_dict_config(self):
        config = {
            "input0": [{"type": "expanddim", "axis": 0}, {"type": "div", "x": 255}],
            "input1": [{"type": "resize", "size": [32, 32]}],
        }
        result = _remove_expanddim(config)
        assert isinstance(result, dict)
        assert all(s["type"] != "expanddim" for s in result["input0"])
        assert len(result["input1"]) == 1

    def test_non_list_non_dict(self):
        assert _remove_expanddim(None) is None
        assert _remove_expanddim("string") == "string"
