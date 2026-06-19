"""CLIP zero-shot classification postprocessing.

Converts a CLIP image embedding into class logits by L2-normalizing the
embedding and taking the dot product with precomputed text (class) embeddings
supplied via the ``text_embedding`` kwarg (shape ``[D, num_classes]``).
"""
from __future__ import annotations

import numpy as np

from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY


@POSTPROCESSING_REGISTRY.register("clip_zero_shot")
class ClipZeroShot:
    """Image-embedding x text-embedding -> class logits.

    Input: image embedding ``[1, D]`` (single model output).
    kwargs: ``text_embedding`` ``[D, num_classes]``.
    Output: logits ``[1, num_classes]``.
    """

    def __init__(self, logit_scale: float = 100.0, **kwargs) -> None:
        self.logit_scale = logit_scale

    def __call__(self, outputs, **kwargs):
        emb = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]

        text_embedding = kwargs.get("text_embedding")
        if text_embedding is None:
            return emb
        text_embedding = np.asarray(text_embedding, dtype=np.float32)

        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-12, norm)
        emb = emb / norm

        logits = self.logit_scale * (emb @ text_embedding)
        return logits
