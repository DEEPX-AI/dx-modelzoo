"""Google Objectron dataset for 6-DoF object pose evaluation (CenterPose).

Reads Objectron **native TFRecord shards** (e.g. ``bottle_test-XXXXX-of-00379``)
with a pure-Python TFRecord + protobuf reader (no tensorflow / crcmod
dependency).  Each record is a ``tf.train.Example`` carrying the encoded image
plus the 3D-box annotation (``point_2d`` = 9 projected keypoints: centroid + 8
cuboid corners).  The reader is category-agnostic — any Objectron category's
``test-`` shards work.
"""
from __future__ import annotations

import os
import struct
from typing import Dict, List, Tuple

import cv2
import numpy as np

from dx_modelzoo.common.dataloader import DatasetBase
from dx_modelzoo.dataset import DATASET_REGISTRY

# ---------------------------------------------------------------------------
# Pure-Python TFRecord reader (no tensorflow / crcmod; CRC not verified)
# ---------------------------------------------------------------------------
# TFRecord layout per record:
#   uint64 length (LE) | uint32 masked-crc32c(length) | bytes data | uint32 crc(data)


def _iter_tfrecord_offsets(path: str):
    """Yield (data_offset, length) for every record in a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            (length,) = struct.unpack("<Q", header)
            f.read(4)  # crc of length
            data_offset = f.tell()
            f.seek(length + 4, os.SEEK_CUR)  # skip payload + crc
            yield data_offset, length


def _read_record(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)


# ---- minimal protobuf wire decoding ---------------------------------------
def _read_varint(buf: bytes, i: int):
    shift = 0
    result = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
    return None, i


def _iter_fields(buf: bytes):
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        if tag is None:
            break
        field = tag >> 3
        wt = tag & 7
        if wt == 0:
            v, i = _read_varint(buf, i)
            yield field, 0, v
        elif wt == 2:
            ln, i = _read_varint(buf, i)
            yield field, 2, buf[i : i + ln]
            i += ln
        elif wt == 5:
            yield field, 5, buf[i : i + 4]
            i += 4
        elif wt == 1:
            yield field, 1, buf[i : i + 8]
            i += 8
        else:
            break


def _parse_example(rec: bytes) -> Dict[str, bytes]:
    """Decode tf.train.Example -> {feature_name: Feature-message bytes}."""
    features = None
    for f, wt, v in _iter_fields(rec):
        if f == 1 and wt == 2:  # Example.features
            features = v
            break
    out: Dict[str, bytes] = {}
    if features is None:
        return out
    for f, wt, v in _iter_fields(features):  # Features.feature (map entries)
        if f != 1 or wt != 2:
            continue
        key = None
        val = None
        for ef, ewt, ev in _iter_fields(v):
            if ef == 1 and ewt == 2:
                key = ev.decode("utf-8", "replace")
            elif ef == 2 and ewt == 2:
                val = ev
        if key is not None and val is not None:
            out[key] = val
    return out


def _feature_list(feature: bytes):
    """Return (kind, list_message_bytes) for a Feature (1=bytes,2=float,3=int64)."""
    for f, wt, v in _iter_fields(feature):
        if f in (1, 2, 3):
            return f, v
    return None, None


def _floats(feature: bytes) -> List[float]:
    _, lst = _feature_list(feature)
    out: List[float] = []
    if lst is None:
        return out
    for f, wt, v in _iter_fields(lst):
        if f == 1 and wt == 2:  # packed
            for i in range(0, len(v), 4):
                out.append(struct.unpack("<f", v[i : i + 4])[0])
        elif f == 1 and wt == 5:  # single
            out.append(struct.unpack("<f", v)[0])
    return out


def _ints(feature: bytes) -> List[int]:
    _, lst = _feature_list(feature)
    out: List[int] = []
    if lst is None:
        return out
    for f, wt, v in _iter_fields(lst):
        if f == 1 and wt == 0:
            out.append(v)
        elif f == 1 and wt == 2:
            i = 0
            while i < len(v):
                x, i = _read_varint(v, i)
                out.append(x)
    return out


def _bytes(feature: bytes) -> bytes:
    _, lst = _feature_list(feature)
    if lst is None:
        return b""
    for f, wt, v in _iter_fields(lst):
        if f == 1 and wt == 2:
            return v
    return b""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_INSTALL_GUIDE = """\
  [Objectron] — Google Objectron native TFRecord shards

  Expected directory containing shards named (one category per eval_path):
      <category>_test-00000-of-00379
      <category>_test-00001-of-00379
      ...
  Each record is a tf.train.Example with image/encoded + point_2d (9 keypoints).
  Source: https://github.com/google-research-datasets/Objectron
"""


@DATASET_REGISTRY.register
class Objectron(DatasetBase):
    """Objectron test split read directly from native TFRecord shards.

    Category-agnostic: indexes every ``*test-*`` shard in ``data_dir``.

    Returns per sample: ``(preprocessed_img, gt_keypoints, (H, W))`` where
    ``gt_keypoints`` is ``float32 [num_instances, 9, 2]`` of normalised (x, y)
    image coordinates (point 0 = centroid, points 1..8 = projected cuboid
    corners).
    """

    def __init__(self, data_dir: str, sample: int = 300, **kwargs) -> None:
        self.ensure_exists(data_dir, _INSTALL_GUIDE)
        super().__init__(data_dir)
        self.sample_cap = int(sample) if sample else 0
        shards = sorted(
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if "test-" in f and not f.endswith(".py")
        )
        # Build a flat index of (shard_path, offset, length), capped at sample.
        self.index: List[Tuple[str, int, int]] = []
        for shard in shards:
            for off, ln in _iter_tfrecord_offsets(shard):
                self.index.append((shard, off, ln))
                if self.sample_cap and len(self.index) >= self.sample_cap:
                    break
            if self.sample_cap and len(self.index) >= self.sample_cap:
                break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple:
        shard, off, ln = self.index[idx]
        rec = _read_record(shard, off, ln)
        d = _parse_example(rec)

        img = cv2.imdecode(np.frombuffer(_bytes(d["image/encoded"]), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image at index {idx} ({shard})")
        H, W = img.shape[:2]

        pts = np.array(_floats(d["point_2d"]), dtype=np.float32)
        # point_2d is [instances * 9 * 3] (x, y, depth); keep (x, y) only.
        if pts.size and pts.size % 3 == 0:
            pts = pts.reshape(-1, 3)[:, :2]
            kpts = pts.reshape(-1, 9, 2) if pts.shape[0] % 9 == 0 else pts[None, :9, :]
        else:
            kpts = np.zeros((0, 9, 2), dtype=np.float32)

        return self.preprocessing(img), kpts.astype(np.float32), (H, W)
