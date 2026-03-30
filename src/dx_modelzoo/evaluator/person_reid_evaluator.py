import time

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from dx_modelzoo.dataset.market1501 import Market1501Dataset
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.session import SessionBase


class PersonReIDEvaluator(EvaluatorBase):
    """Person Re-Identification Evaluator using CMC and mAP.

    Extracts embeddings from gallery and query sets, computes cosine
    distance matrix, and evaluates Rank-1 / mAP following Market-1501 protocol.

    Args:
        session (SessionBase): runtime session.
        dataset (Market1501Dataset): Market-1501 dataset.
    """

    def __init__(self, session: SessionBase, dataset: Market1501Dataset):
        super().__init__(session, dataset, workers=4)
        self.dataset: Market1501Dataset
        self._gallery_embeddings: list = []
        # ReID requires two-pass (gallery→query), force sync mode
        self._use_async = False

    def init_metrics(self) -> dict:
        return {"embeddings": [], "phase": "gallery"}

    def extract_inputs(self, batch_data) -> torch.Tensor:
        image, idx = batch_data
        return image

    def process_batch_result(self, batch_data, output, metrics_state: dict) -> dict:
        emb = self._to_numpy(output)
        metrics_state["embeddings"].append(emb)
        return metrics_state

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        gallery_embs = np.vstack(self._gallery_embeddings)
        query_embs = np.vstack(metrics_state["embeddings"])

        gallery_ids = self.dataset.gallery_ids
        gallery_cams = self.dataset.gallery_cams
        query_ids = self.dataset.query_ids
        query_cams = self.dataset.query_cams

        # Filter out pid=0 (junk) from gallery
        valid_gallery = gallery_ids > 0
        gallery_embs = gallery_embs[valid_gallery]
        gallery_ids = gallery_ids[valid_gallery]
        gallery_cams = gallery_cams[valid_gallery]

        # L2 normalize
        gallery_embs = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-10)
        query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)

        # Cosine distance
        dist_mat = 1 - np.dot(query_embs, gallery_embs.T)

        cmc, mAP = self._eval_market1501(dist_mat, query_ids, gallery_ids, query_cams, gallery_cams)

        rank1 = cmc[0]
        total_images = len(self.dataset.gallery) + len(self.dataset.query)
        avg_fps = total_images / self.total_inference_time if self.total_inference_time > 0 else 0.0

        print(f"Rank-1: {rank1:.4f}")
        print(f"mAP: {mAP:.4f}")
        print(f"Average FPS: {avg_fps:.2f}")
        logger.success(f"@JSON <Rank1:{rank1:.4f}; mAP:{mAP:.4f}; Average FPS:{avg_fps:.2f}>")

        return {
            "performance": [rank1, mAP],
            "fps": avg_fps,
        }

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        phase = metrics_state.get("phase", "gallery")
        n = len(metrics_state["embeddings"])
        return f"ReID ({phase}) | Embs:{n} FPS:{current_fps:.1f}"

    def _eval_sync(self) -> dict:
        """Override to do two-pass: gallery then query."""
        # Phase 1: Gallery embeddings
        self.dataset.set_mode("gallery")
        loader = self.make_loader()
        self._gallery_embeddings = []

        pbar = tqdm(enumerate(loader), total=len(loader), desc="Gallery")
        for batch_idx, batch_data in pbar:
            start_time = time.time()
            inputs = self.extract_inputs(batch_data)
            output = self.session.run(inputs)
            inference_time = time.time() - start_time
            self.recent_inference_times.append(inference_time)
            self.total_inference_time += inference_time

            if not getattr(self, "lazy_postprocessing", False):
                output = self.postprocessing(output)
            emb = self._to_numpy(output)
            self._gallery_embeddings.append(emb)

            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0
            pbar.desc = f"Gallery | Embs:{batch_idx + 1} FPS:{current_fps:.1f}"
        pbar.close()

        # Phase 2: Query embeddings
        self.dataset.set_mode("query")
        loader = self.make_loader()
        metrics_state = self.init_metrics()
        metrics_state["phase"] = "query"

        pbar = tqdm(enumerate(loader), total=len(loader), desc="Query")
        for batch_idx, batch_data in pbar:
            start_time = time.time()
            inputs = self.extract_inputs(batch_data)
            output = self.session.run(inputs)
            inference_time = time.time() - start_time
            self.recent_inference_times.append(inference_time)
            self.total_inference_time += inference_time

            if not getattr(self, "lazy_postprocessing", False):
                output = self.postprocessing(output)
            metrics_state = self.process_batch_result(batch_data, output, metrics_state)

            if len(self.recent_inference_times) > 0:
                current_fps = len(self.recent_inference_times) / sum(self.recent_inference_times)
            else:
                current_fps = 0.0
            pbar.desc = self.format_progress_desc(metrics_state, current_fps)
        pbar.close()

        return self.compute_final_metrics(metrics_state)

    @staticmethod
    def _eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Market-1501 evaluation protocol."""
        num_q = distmat.shape[0]

        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []

        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = ~remove

            raw_cmc = matches[q_idx][keep]
            if not np.any(raw_cmc):
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])

            # AP
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
            tmp_cmc = raw_cmc * precision
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        all_cmc = np.array(all_cmc, dtype=np.float32)
        cmc = np.mean(all_cmc, axis=0)
        mAP = float(np.mean(all_AP))

        return cmc, mAP

    @staticmethod
    def _to_numpy(output) -> np.ndarray:
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        return np.atleast_2d(np.squeeze(output))
