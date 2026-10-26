"""Anomaly-detection evaluator for MVTec AD (PatchCore & EfficientAD).

Reports image-level AUROC (and optional pixel-level AUROC), averaged over the
evaluated categories -- the standard MVTec AD protocol.

Three scoring back-ends are supported, selected via ``evaluator.options.method``
(or the model YAML ``appendix.method``):

* ``patchcore`` -- The model is a WideResNet-style feature extractor emitting
  two feature maps (``feat_layer2`` [1,512,28,28] and ``feat_layer3``
  [1,1024,14,14]).  A per-category memory bank is built from ``train/good``
  patch features (local 3x3 aggregation + concat, random coreset subsample).
  The image score is the maximum patch-to-bank nearest-neighbour distance.

* ``efficientad`` -- Requires the teacher/student/autoencoder trio.  Whichever
  of the three model YAMLs is evaluated, the evaluator locates and loads all
  three sibling models and combines them.  Teacher features are normalised by
  per-channel mean/std computed on ``train/good``; the anomaly map is the mean
  of the student-teacher regression error and the autoencoder-student error,
  each scaled by a ``train/good`` quantile.  Image score = max of the map.
* ``efficientad_component`` -- Evaluates one EfficientAD component in
  isolation using a PatchCore-style feature memory bank.  This mode is
  intended for workers that provide only one model artifact.

Reference implementations:
  PatchCore  -- https://github.com/amazon-science/patchcore-inspection
  EfficientAD -- anomalib / https://github.com/nelson1425/EfficientAD
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from dx_modelzoo.session import SessionBase

_EA_NAMES = {
    "teacher": "efficientad-m-teacher_256x256",
    "student": "efficientad-m-student_256x256",
    "autoencoder": "efficientad-m-autoencoder_256x256",
}


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUROC; robust to ties. NaN if single-class."""
    from scipy.stats import rankdata

    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@EVALUATOR_REGISTRY.register("anomaly_detection")
class AnomalyDetectionEvaluator(EvaluatorBase):
    """MVTec AD image/pixel AUROC evaluator (PatchCore & EfficientAD modes)."""

    # -- configurable via YAML appendix / evaluator.options -------------------
    method: str = "patchcore"  # "patchcore" | "efficientad" | "efficientad_component"
    pixel_auroc: bool = True  # also compute pixel-level AUROC
    bank_size: int = 16384  # PatchCore coreset (random) size per category
    map_size: int = 0  # anomaly-map size for pixel AUROC (0 = model input H)
    ea_quantile: float = 0.9  # EfficientAD train-map normalization quantile

    def __init__(self, session: SessionBase, dataset: DatasetBase, **kwargs) -> None:
        super().__init__(session, dataset, **kwargs)
        self._banks: Dict[str, Any] = {}  # patchcore: category -> bank tensor
        self._ea_sessions: Dict[str, SessionBase] = {}
        self._ea_stats: Dict[str, Dict[str, Any]] = {}  # category -> stats
        self._records: List[Tuple[str, int, float]] = []  # (category, label, image_score)
        self._pix_scores: Dict[str, List[np.ndarray]] = {}
        self._pix_labels: Dict[str, List[np.ndarray]] = {}
        self._prepared = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _to_input(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, 0)
        return arr

    def _model_hw(self) -> Tuple[int, int]:
        shp = self.model_spec.get("input_resolution")
        if shp and len(shp) >= 2:
            return int(shp[0]), int(shp[1])
        return 224, 224

    # ---------------------------------------------------- PatchCore feature ops
    def _patchcore_patches(self, outputs: List[np.ndarray]):
        """Local-aggregated, spatially-aligned, concatenated patch features.

        Returns (patches [H*W, C] tensor, (H, W)).
        """
        import torch
        import torch.nn.functional as F

        embeds = []
        for arr in outputs:
            t = torch.from_numpy(np.ascontiguousarray(arr)).float()
            t = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)  # local neighbourhood
            embeds.append(t)
        target = embeds[0].shape[-2:]
        embeds = [
            e if e.shape[-2:] == target else F.interpolate(e, size=target, mode="bilinear", align_corners=False)
            for e in embeds
        ]
        emb = torch.cat(embeds, dim=1)  # [1, C, H, W]
        c, h, w = emb.shape[1], emb.shape[2], emb.shape[3]
        patches = emb.reshape(1, c, h * w).permute(0, 2, 1).reshape(h * w, c)
        return patches, (h, w)

    def _build_bank(self, category: str):
        import torch

        feats = []
        files = self.dataset.train_files(category)
        for p in files:
            img = self.dataset.preprocess_path(p)
            out = self.session.run(self._to_input(img))
            patches, _ = self._patchcore_patches(out)
            feats.append(patches)
        allp = torch.cat(feats, dim=0)
        n = allp.shape[0]
        if self.bank_size and n > self.bank_size:
            idx = torch.randperm(n)[: self.bank_size]
            allp = allp[idx]
        logger.info(f"[patchcore] category={category}: memory bank {tuple(allp.shape)} from {len(files)} imgs")
        return allp.contiguous()

    def _score_patchcore(self, output, category: str):
        import torch

        patches, (h, w) = self._patchcore_patches(output)
        bank = self._banks[category]
        dists = torch.cdist(patches, bank)  # [HW, B]
        min_d = dists.min(dim=1).values  # [HW]
        img_score = float(min_d.max().item())
        amap = min_d.reshape(h, w).cpu().numpy()
        return img_score, amap

    # --------------------------------------------------------- EfficientAD ops
    def _ea_setup_sessions(self) -> None:
        from dx_modelzoo.session.factory import create_session

        main = Path(self.session.path)
        suffix = main.suffix
        root = main.parent.parent
        dxnn_suffix = ""
        if suffix == ".dxnn":
            for v in _EA_NAMES.values():
                if main.stem.startswith(v):
                    dxnn_suffix = main.stem[len(v) :]  # e.g. "_q-lite"
                    break
        missing = []
        for key, v in _EA_NAMES.items():
            d = root / v
            mp = d / (f"{v}.onnx" if suffix == ".onnx" else f"{v}{dxnn_suffix}.dxnn")
            if not mp.exists():
                missing.append(f"{key}: {mp}")
                continue
            self._ea_sessions[key] = create_session(str(mp))
            logger.info(f"[efficientad] loaded {key}: {mp}")
        if missing:
            raise FileNotFoundError(
                "[efficientad] standalone evaluation requires teacher, student, and autoencoder "
                "artifacts in the same model root. Use method 'efficientad_component' for "
                "per-model workers or provide all sibling artifacts. Missing: "
                f"{'; '.join(missing)}"
            )

    def _ea_raw_maps(self, img: np.ndarray):
        """Return (st_map [h,w], ae_map [h,w]) unnormalized, given a train stat mean/std."""
        inp = self._to_input(img)
        teacher = np.asarray(self._ea_sessions["teacher"].run(inp)[0])[0]  # [384,h,w]
        student = np.asarray(self._ea_sessions["student"].run(inp)[0])[0]  # [768,h,w]
        ae = np.asarray(self._ea_sessions["autoencoder"].run(inp)[0])[0]  # [384,h,w]
        return teacher, student, ae

    def _ea_build_stats(self, category: str) -> Dict[str, Any]:
        """Calibrate the EfficientAD trio on this category's train/good images.

        The teacher / student / autoencoder ONNX files were exported separately
        and their feature channels are *not* in a common ordering (verified: the
        student's teacher-branch is spatially uncorrelated with the teacher
        output channel-for-channel).  We therefore recover, from anomaly-free
        train images, a per-pixel ridge-linear map that predicts the student's
        teacher-branch from the teacher features and its autoencoder-branch from
        the autoencoder features.  On a normal image the student is well
        predicted (small residual); an anomaly breaks the learned relation and
        the squared, residual-variance-normalised error spikes.  Maps are scaled
        by a train quantile and combined 50/50 (EfficientAD map-normalisation).
        """
        outs = []  # cached (teacher, student, ae) as float16
        for pth in self.dataset.train_files(category):
            img = self.dataset.preprocess_path(pth)
            teacher, student, ae = self._ea_raw_maps(img)
            outs.append((teacher.astype(np.float16), student.astype(np.float16), ae.astype(np.float16)))
        if not outs:
            raise ValueError(f"[efficientad] no train/good images for category {category!r}")

        c = outs[0][0].shape[0]  # teacher channels (384)

        def _flat(idx, lo=None, hi=None):
            mats = []
            for tup in outs:
                v = tup[idx].astype(np.float32)
                if lo is not None:
                    v = v[lo:hi]
                mats.append(v.reshape(v.shape[0], -1).T)
            return np.concatenate(mats, axis=0)

        teacher_f = _flat(0)
        ae_f = _flat(2)
        student_st_f = _flat(1, 0, c)
        student_ae_f = _flat(1, c, 2 * c)

        w_st, rv_st = self._fit(teacher_f, student_st_f)
        w_ae, rv_ae = self._fit(ae_f, student_ae_f)
        del teacher_f, ae_f, student_st_f, student_ae_f

        stats = {"w_st": w_st, "rv_st": rv_st, "w_ae": w_ae, "rv_ae": rv_ae, "c": c, "q_st": 1.0, "q_ae": 1.0}
        st_vals, ae_vals = [], []
        for teacher, student, ae in outs:
            rst, rae = self._ea_resid_maps(
                teacher.astype(np.float32), student.astype(np.float32), ae.astype(np.float32), stats
            )
            st_vals.append(rst.reshape(-1))
            ae_vals.append(rae.reshape(-1))
        stats["q_st"] = max(float(np.quantile(np.concatenate(st_vals), self.ea_quantile)), 1e-6)
        stats["q_ae"] = max(float(np.quantile(np.concatenate(ae_vals), self.ea_quantile)), 1e-6)
        logger.info(
            f"[efficientad] category={category}: calibrated on {len(outs)} imgs "
            f"(q_st={stats['q_st']:.4f}, q_ae={stats['q_ae']:.4f})"
        )
        return stats

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2):
        """Ridge least squares y ~ [x, 1]; returns (weights, residual variance)."""
        xb = np.concatenate([x, np.ones((x.shape[0], 1), np.float32)], axis=1)
        xtx = xb.T @ xb + ridge * np.eye(xb.shape[1], dtype=np.float32)
        w = np.linalg.solve(xtx, xb.T @ y).astype(np.float32)
        rv = (y - xb @ w).var(axis=0) + 1e-9
        return w, rv.astype(np.float32)

    @staticmethod
    def _ea_resid_maps(teacher: np.ndarray, student: np.ndarray, ae: np.ndarray, stats: Dict[str, Any]):
        c = stats["c"]
        hw = teacher.shape[1]
        tp = teacher.reshape(c, -1).T
        ap = ae.reshape(c, -1).T
        tb = np.concatenate([tp, np.ones((tp.shape[0], 1), np.float32)], axis=1)
        ab = np.concatenate([ap, np.ones((ap.shape[0], 1), np.float32)], axis=1)
        st_pred = tb @ stats["w_st"]
        ae_pred = ab @ stats["w_ae"]
        st_res = (((student[:c].reshape(c, -1).T - st_pred) ** 2) / stats["rv_st"]).mean(axis=1)
        ae_res = (((student[c : 2 * c].reshape(c, -1).T - ae_pred) ** 2) / stats["rv_ae"]).mean(axis=1)
        return st_res.reshape(hw, hw), ae_res.reshape(hw, hw)

    def _score_efficientad(self, img: np.ndarray, category: str):
        stats = self._ea_stats[category]
        teacher, student, ae = self._ea_raw_maps(img)
        st_map, ae_map = self._ea_resid_maps(teacher, student, ae, stats)
        combined = 0.5 * (st_map / stats["q_st"]) + 0.5 * (ae_map / stats["q_ae"])
        return float(combined.max()), combined

    # ----------------------------------------------------------- preparation
    def _prepare(self) -> None:
        if self._prepared:
            return
        cats = self.dataset.categories
        if self.method in {"patchcore", "efficientad_component"}:
            for cat in cats:
                self._banks[cat] = self._build_bank(cat)
        elif self.method == "efficientad":
            self._ea_setup_sessions()
            for cat in cats:
                self._ea_stats[cat] = self._ea_build_stats(cat)
        else:
            raise ValueError(f"Unknown anomaly method: {self.method!r}")
        self._prepared = True

    # ------------------------------------------------------- EvaluatorBase API
    def init_metrics(self) -> dict:
        self._records = []
        self._pix_scores = {}
        self._pix_labels = {}
        self._prepare()
        return {"count": 0}

    def extract_inputs(self, batch_data: Tuple) -> np.ndarray:
        return self._to_input(batch_data[0])

    def _accumulate_pixels(self, category: str, amap: np.ndarray, mask_path: str, label: int) -> None:
        if not self.pixel_auroc:
            return
        size = self.map_size or self._model_hw()[0]
        amap_r = cv2.resize(amap.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        if label == 0 or not mask_path:
            gt = np.zeros((size, size), dtype=np.uint8)
        else:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                gt = np.zeros((size, size), dtype=np.uint8)
            else:
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                gt = (m > 0).astype(np.uint8)
        self._pix_scores.setdefault(category, []).append(amap_r.reshape(-1))
        self._pix_labels.setdefault(category, []).append(gt.reshape(-1))

    def process_batch_result(self, batch_data: Tuple, output: Any, metrics_state: dict) -> dict:
        img, label, category, mask_path = batch_data[0], int(batch_data[1]), batch_data[2], batch_data[3]

        if self.method in {"patchcore", "efficientad_component"}:
            img_score, amap = self._score_patchcore(output, category)
        else:
            img_score, amap = self._score_efficientad(img, category)

        self._records.append((category, label, img_score))
        self._accumulate_pixels(category, amap, mask_path, label)
        metrics_state["count"] += 1
        return metrics_state

    def _mean_over_categories(self):
        cats = self.dataset.categories
        img_aurocs, pix_aurocs = [], []
        for cat in cats:
            labels = np.array([l for (c, l, s) in self._records if c == cat])
            scores = np.array([s for (c, l, s) in self._records if c == cat])
            a = _auroc(labels, scores)
            if not np.isnan(a):
                img_aurocs.append(a)
            logger.info(f"  [{cat}] image-AUROC={a:.4f} (n={len(labels)})")
            if self.pixel_auroc and cat in self._pix_scores:
                ps = np.concatenate(self._pix_scores[cat])
                pl = np.concatenate(self._pix_labels[cat])
                pa = _auroc(pl, ps)
                if not np.isnan(pa):
                    pix_aurocs.append(pa)
                logger.info(f"  [{cat}] pixel-AUROC={pa:.4f}")
        img_mean = float(np.mean(img_aurocs)) if img_aurocs else 0.0
        pix_mean = float(np.mean(pix_aurocs)) if pix_aurocs else float("nan")
        return img_mean, pix_mean

    def compute_final_metrics(self, metrics_state: dict) -> dict:
        img_mean, pix_mean = self._mean_over_categories()
        n = metrics_state["count"]
        fps = n / self.total_inference_time if self.total_inference_time > 0 else 0.0
        names = ["image_AUROC"]
        values = [img_mean * 100.0]
        if self.pixel_auroc and not np.isnan(pix_mean):
            names.append("pixel_AUROC")
            values.append(pix_mean * 100.0)
        return self._finalize(metric_names=names, metric_values=values, fps=fps)

    def format_progress_desc(self, metrics_state: dict, current_fps: float) -> str:
        return f"MVTecAD[{self.method}] | imgs:{metrics_state.get('count', 0)} FPS:{current_fps:.1f}"
