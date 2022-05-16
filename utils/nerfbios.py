import json
import logging
import os
from typing import Any, Callable, List, Text, Tuple, Union

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def compute_pcks_per_ratio(kps, pred_kps, img_wh, ratios):
    # kps, pred_kps: (N, J, 3)
    valid_mask = kps[..., -1:] != 0
    kps = kps[..., :2][valid_mask.repeat(2, axis=-1)].reshape(-1, 2)
    pred_kps = pred_kps[..., :2][valid_mask.repeat(2, axis=-1)].reshape(-1, 2)

    dists = np.linalg.norm(kps - pred_kps, axis=-1)
    threshes = np.array(ratios) * max(img_wh)
    stats = dists[:, None] < threshes[None, :]

    pcks = stats.sum(0) / stats.shape[0]
    return np.stack([pcks, ratios], axis=-1)
    
    
def get_kp_xv_ids(data_dir: str, num_xv_steps: int = 10):
    train_ids, val_ids, is_multiview = _load_dataset_ids(data_dir)
    train_ids_xv = _strided_subset(train_ids, num_xv_steps, clip_ok=True)
    idxs_xv = [train_ids.index(id) for id in train_ids_xv]
    return idxs_xv


def get_xv_ids(data_dir: str, num_xv_steps: int = 4):
    train_ids, val_ids, is_multiview = _load_dataset_ids(data_dir)
    train_ids_xv = _strided_subset(train_ids, num_xv_steps, clip_ok=True)
    idxs_xv = [train_ids.index(id) for id in train_ids_xv]
    return idxs_xv


def _load_dataset_ids(
    data_dir: str,
) -> Tuple[List[str], List[str], bool]:
    """Loads dataset IDs."""
    dataset_json_path = os.path.join(data_dir, "dataset.json")
    LOGGER.info("*** Loading dataset IDs from %s", dataset_json_path)
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
        train_ids = dataset_json["train_ids"]
        val_ids = dataset_json["val_ids"]
        is_multiview = dataset_json.get("is_multiview", False)
    train_ids = [str(i) for i in train_ids]
    val_ids = [str(i) for i in val_ids]
    return train_ids, val_ids, is_multiview


def _strided_subset(sequence, count, clip_ok=False):
    """Returns a strided subset of a list."""
    num_items = len(sequence)
    if count:
        if count > num_items:
            if not clip_ok:
                raise ValueError(
                    f"Expect count <= num_items, got {count} > {num_items}."
                )
            else:
                return sequence
        end_idx = max(0, num_items - 1)
        num_strides = max(1, count - 1)
        stride = max(1, end_idx // num_strides)
        return sequence[::stride]
    return sequence


if __name__ == "__main__":
    xv_ids = get_xv_ids(
        data_dir="/home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v2/mochi-high-five_0-180-1_aligned_gq90_bk10"
    )
    print ("xv_ids", xv_ids)
