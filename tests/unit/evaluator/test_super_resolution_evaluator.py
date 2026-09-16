"""Tests for dx_modelzoo.evaluator.super_resolution_evaluator (E2E)."""

import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator.super_resolution_evaluator import SuperResolutionEvaluator, _reassemble_tiles, _tile_image
from dx_modelzoo.preprocessing import PreprocessingPipeline
from dx_modelzoo.session import SessionBase


class FakeSession(SessionBase):
    def __init__(self):
        super().__init__("/fake.onnx")

    def run(self, inputs):
        # SR output: same as input (identity)
        return [inputs]


class FakeDataset(DatasetBase):
    def validate(self) -> None:  # ponytail: test double, nothing to validate
        pass

    def __init__(self):
        super().__init__("/fake")
        self.scale = 2

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        lr = np.random.rand(3, 16, 16).astype(np.float32)
        hr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        return lr, hr


class TestSuperResolutionEvaluator:
    def _make(self):
        ev = SuperResolutionEvaluator(FakeSession(), FakeDataset())
        ev.set_postprocessing(lambda outputs, **kw: outputs[0] if isinstance(outputs, list) else outputs)
        return ev

    def test_init_metrics(self):
        ev = self._make()
        metrics = ev.init_metrics()
        assert metrics["total_psnr"] == 0.0

    def test_eval_end_to_end(self):
        ev = self._make()
        result = ev.eval()
        assert isinstance(result, dict)

    def test_stride_13_tiles_reassemble_without_overlap_seams(self):
        image = np.arange(30 * 30, dtype=np.float32).reshape(1, 30, 30)

        tiles = _tile_image(image, tile_size=17, stride=13)
        reconstructed = _reassemble_tiles(tiles, image.shape[-2:], upscale_factor=1)

        assert [(y, x) for y, x, _ in tiles] == [(0, 0), (0, 13), (13, 0), (13, 13)]
        np.testing.assert_array_equal(reconstructed, image)

    def test_tiled_inference_uses_base_eval_hooks(self):
        ev = self._make()
        ev.upscale_factor = 1
        ev.set_preprocessing(PreprocessingPipeline([{"type": "centercrop", "height": 17, "width": 17}]))
        image = np.zeros((1, 30, 30), dtype=np.float32)

        tiled_outputs = ev.run_inference(image, (image, image))
        processed = ev.postprocess_output(tiled_outputs, (image, image))

        assert len(tiled_outputs) == 4
        np.testing.assert_array_equal(processed, image)
