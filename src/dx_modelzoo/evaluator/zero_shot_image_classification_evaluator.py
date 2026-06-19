from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase


@EVALUATOR_REGISTRY.register("zero_shot_image_classification")
class ZeroShotImageClassificationEvaluator(EvaluatorBase):
    """Zero-shot Image Classification Evaluator (CLIP)."""

    def __init__(
        self,
        session: SessionBase,
        dataset: DatasetBase,
        zero_shot_text_embedding: Optional[str] = None,
        clip_model_name: Optional[str] = None,
        clip_pretrained: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(session, dataset, workers=12, **kwargs)
        self.zero_shot_text_embedding = zero_shot_text_embedding
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.zeroshot_text_embedding_weight = None

    @staticmethod
    def _build_text_embedding(model_name: str, pretrained: str) -> np.ndarray:
        import open_clip
        import torch
        from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model '{}'  (pretrained='{}') on {}", model_name, pretrained, device)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        logger.info("CLIP model loaded. Building text embeddings for {} classes...", len(IMAGENET_CLASSNAMES))

        with torch.no_grad():
            text_embeddings = []
            for classname in tqdm(IMAGENET_CLASSNAMES, desc="Building text embeddings", unit="class"):
                texts = [template(classname) for template in OPENAI_IMAGENET_TEMPLATES]
                texts_tokenized = tokenizer(texts).to(device)
                class_embedding = model.encode_text(texts_tokenized)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                class_embedding = class_embedding.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_embeddings.append(class_embedding)
            text_embeddings = torch.stack(text_embeddings, dim=1)

        logger.info("Text embeddings built -- shape: {}", text_embeddings.shape)
        return text_embeddings.cpu().numpy()

    def _accuracy(self, output: np.ndarray, target: np.ndarray, topk=(1, 5)):
        maxk = max(topk)
        pred = np.argsort(-output, axis=1)[:, :maxk]
        correct = pred == target.reshape(-1, 1)
        return [float(correct[:, :k].sum()) for k in topk]

    def init_metrics(self) -> dict:
        if self.zero_shot_text_embedding:
            logger.info("Loading pre-computed text embeddings from '{}'" , self.zero_shot_text_embedding)
            self.zeroshot_text_embedding_weight = np.load(self.zero_shot_text_embedding)
            logger.info("Text embeddings loaded -- shape: {}", self.zeroshot_text_embedding_weight.shape)
        elif self.clip_model_name and self.clip_pretrained:
            self.zeroshot_text_embedding_weight = self._build_text_embedding(self.clip_model_name, self.clip_pretrained)
        else:
            logger.warning(
                "No 'zero_shot_text_embedding' or 'clip_model_name'/'clip_pretrained' specified. "
                "Text embedding weight will be None."
            )
        return {"correct_top1": 0.0, "correct_top5": 0.0, "current_count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        images, labels = batch_data
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = np.expand_dims(images, 0)
        return images

    def _build_postprocessing_context(self, batch_data) -> dict:
        return {"text_embedding": self.zeroshot_text_embedding_weight}

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        images, labels = batch_data
        logits = np.asarray(output)
        labels_np = np.atleast_1d(np.asarray(labels))
        acc1, acc5 = self._accuracy(logits, labels_np, topk=(1, 5))
        batch_size = logits.shape[0]
        metrics_state["correct_top1"] += acc1
        metrics_state["correct_top5"] += acc5
        metrics_state["current_count"] += batch_size
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        n = metrics_state["current_count"]
        top1 = (metrics_state["correct_top1"] / n * 100) if n > 0 else 0.0
        top5 = (metrics_state["correct_top5"] / n * 100) if n > 0 else 0.0
        avg_fps = n / self.total_inference_time if self.total_inference_time > 0 else 0.0
        return self._finalize(
            metric_names=["Top1 Accuracy", "Top5 Accuracy"],
            metric_values=[top1, top5],
            fps=avg_fps,
        )

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        n = metrics_state["current_count"]
        if n == 0:
            return "ImageNet | Initializing..."
        t1 = metrics_state["correct_top1"] / n
        t5 = metrics_state["correct_top5"] / n
        return f"ImageNet | Top1:{t1:.2f} Top5:{t5:.2f} Current_FPS:{current_fps:.1f}"
