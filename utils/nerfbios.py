import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from typing import Any, Callable, List, Optional, Text, Tuple, Union

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


UINT8_MAX = 255
UINT16_MAX = 65535


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to a uint8 array."""
    image = np.array(image)
    if image.dtype == np.uint8:
        return image
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(
            f"Input image should be a floating type but is of type "
            f"{image.dtype!r}"
        )
    return (image * UINT8_MAX).clip(0.0, UINT8_MAX).astype(np.uint8)


def random_color(num_items, seed=0, with_bkgd=False):
    colors = np.random.RandomState(seed).randint(
        0, 255, size=(num_items, 3), dtype=np.uint8
    )
    if with_bkgd:
        # White bkgd.
        colors = np.concatenate([colors, np.full_like(colors[:1], 255)], axis=0)
    return colors


KP_NAMES = [
    "LFrontPaw",
    "LFrontWrist",
    "LFrontElbow",
    "LRearPaw",
    "LRearWrist",
    "LRearElbow",
    "RFrontPaw",
    "RFrontWrist",
    "RFrontElbow",
    "RRearPaw",
    "RRearWrist",
    "RRearElbow",
    "TailStart",
    "TailEnd",
    "LEar",
    "REar",
    "Nose",
    "Chin",
    "LEarTip",
    "REarTip",
    "LEye",
    "REye",
    "Withers",
    "Throat",
]
KP_INDS = {kp_name: kp_id for kp_id, kp_name in enumerate(KP_NAMES)}
KP_PALETTE_MAP = {
    "LFrontPaw": (0, 255, 0),
    "LFrontWrist": (63, 255, 0),
    "LFrontElbow": (127, 255, 0),
    "LRearPaw": (0, 0, 255),
    "LRearWrist": (0, 63, 255),
    "LRearElbow": (0, 127, 255),
    "RFrontPaw": (255, 255, 0),
    "RFrontWrist": (255, 191, 0),
    "RFrontElbow": (255, 127, 0),
    "RRearPaw": (0, 255, 255),
    "RRearWrist": (0, 255, 191),
    "RRearElbow": (0, 255, 127),
    "TailStart": (0, 0, 0),
    "TailEnd": (0, 0, 0),
    "LEar": (255, 0, 170),
    "REar": (255, 0, 170),
    "Nose": (255, 0, 170),
    "Chin": (255, 0, 170),
    "LEarTip": (255, 0, 170),
    "REarTip": (255, 0, 170),
    "LEye": (255, 0, 170),
    "REye": (255, 0, 170),
    "Withers": (255, 0, 170),
    "Throat": (255, 0, 170),
}


class cached_property(object):
    """Property that caches. Assume dependent attributes won't change."""

    def __init__(self, fget):
        self.fget = fget
        self.__module__ = fget.__module__
        self.__name__ = fget.__name__
        self.__doc__ = fget.__doc__

        self.__cache_key = "__result_cache_{0}_{1}".format(
            fget.__name__, id(fget)
        )
        self.__mutex = defaultdict(threading.Lock)

    def __get__(self, instance, owner):
        with self.__mutex[instance]:
            if instance is None:
                return self.fget

            v = getattr(instance, self.__cache_key, None)
            if v is not None:
                return v

            v = self.fget(instance)
            assert v is not None
            setattr(instance, self.__cache_key, v)
            return v

class Skeleton(object):
    _anonymous_kp_name = "ANONYMOUS KP"

    def __init__(
        self,
        parents: list,
        kp_names: Optional[list] = None,
        palette: Optional[list] = None,
    ):
        if kp_names is not None:
            assert len(parents) == len(kp_names)
            if palette is not None:
                assert len(kp_names) == len(palette)

        self._parents = parents
        self._kp_names = (
            kp_names
            if kp_names is not None
            else [self._anonymous_kp_name] * self.num_kps
        )
        self._palette = palette

    @property
    def parents(self):
        return self._parents

    @property
    def kp_names(self):
        return self._kp_names

    @cached_property
    def palette(self):
        if self._palette is not None:
            return self._palette

        if self.kp_names[0] != self._anonymous_kp_name and all(
            [kp_name in KP_PALETTE_MAP for kp_name in self.kp_names]
        ):
            return [KP_PALETTE_MAP[kp_name] for kp_name in self.kp_names]

        palette = np.zeros((self.num_kps, 3), dtype=np.uint8)
        left_mask = np.array(
            [
                len(re.findall(r"^(\w+ |)L\w+$", kp_name)) > 0
                for kp_name in self._kp_names
            ],
            dtype=np.bool,
        )
        palette[left_mask] = (255, 0, 0)
        return [tuple(color.tolist()) for color in palette]

    @cached_property
    def num_kps(self):
        return len(self._parents)

    @cached_property
    def root_idx(self):
        return self._parents.index(-1)

    @cached_property
    def bones(self):
        return np.stack([list(range(self.num_kps)), self.parents]).T.tolist()

    @cached_property
    def non_root_bones(self):
        return np.delete(self.bones.copy(), self.root_idx, axis=0)

    @cached_property
    def non_root_palette(self):
        return np.delete(self.palette.copy(), self.root_idx, axis=0).tolist()


STANFORDX_SKEL = Skeleton(
    parents=[
        1,
        2,
        22,
        4,
        5,
        12,
        7,
        8,
        22,
        10,
        11,
        12,
        -1,
        12,
        20,
        21,
        17,
        23,
        14,
        15,
        16,
        16,
        12,
        22,
    ],
    kp_names=[
        "LFrontPaw",
        "LFrontWrist",
        "LFrontElbow",
        "LRearPaw",
        "LRearWrist",
        "LRearElbow",
        "RFrontPaw",
        "RFrontWrist",
        "RFrontElbow",
        "RRearPaw",
        "RRearWrist",
        "RRearElbow",
        "TailStart",
        "TailEnd",
        "LEar",
        "REar",
        "Nose",
        "Chin",
        "LEarTip",
        "REarTip",
        "LEye",
        "REye",
        "Withers",
        "Throat",
    ],
)
SKEL_MAP = {"stanfordx": STANFORDX_SKEL, None: None}


def visualize_kps(
    kps: np.ndarray,
    image: np.ndarray,
    *,
    skel = "stanfordx",
    kp_radius: int = 4,
    bone_thickness: int = 3,
) -> np.ndarray:
    """Visualize 2D keypoints.

    Args:
        kps (np.ndarray): an array of shape (J, 3) for keypoints. Expect the
            last column to be the visibility in [0, 1].
        image (np.ndarray): a RGB image of shape (H, W, 3) that aligns with
            `kps` in uint8.
        skel (Skeleton): a skeleton definition object. If None, assume
            non-skeletal structure and no draw no bones. Default:
            STANFORDX_SKEL.
        kp_radius (int): the radius of `kps` for visualization. Default: 4.
        bone_thickness (int): the thickness of bones connecting `kps` for
            visualization. Default: 3.

    Returns:
        np.ndarray: a keypoint visualzation image of shape (H, W, 3) in uint8.
    """
    skel = SKEL_MAP[skel]
    assert (skel is not None and kps.shape == (skel.num_kps, 3)) or skel is None

    kps = np.array(kps)
    image = image_to_uint8(image)

    H, W = image.shape[:2]
    canvas = image.copy()

    valid_mask = (
        (kps[:, -1] != 0)
        & (kps[:, 0] >= 0)
        & (kps[:, 0] < W)
        & (kps[:, 1] >= 0)
        & (kps[:, 1] < H)
    )

    if skel is not None:
        palette = skel.non_root_palette
        bones = skel.non_root_bones
        for color, (j, p) in zip(palette, bones):
            # Skip invisible keypoints.
            if (~valid_mask[[j, p]]).any():
                continue

            kp_p, kp_j = kps[p, :2], kps[j, :2]
            kp_mid = (kp_p + kp_j) / 2
            bone_length = np.linalg.norm(kp_j - kp_p)
            bone_angle = (
                (np.arctan2(kp_j[1] - kp_p[1], kp_j[0] - kp_p[0])) * 180 / np.pi
            )
            polygon = cv2.ellipse2Poly(
                (int(kp_mid[0]), int(kp_mid[1])),
                (int(bone_length / 2), bone_thickness),
                int(bone_angle),
                arcStart=0,
                arcEnd=360,
                delta=5,
            )
            cv2.fillConvexPoly(canvas, polygon, color, lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

    # Same random colors for fixed number of keypoints.
    combined = canvas.copy()
    palette = (
        skel.palette if skel else tuple(random_color(len(kps)).tolist())
    )
    for color, kp, valid in zip(palette, kps, valid_mask):
        # Skip invisible keypoints.
        if not valid:
            continue

        cv2.circle(
            combined,
            (int(kp[0]), int(kp[1])),
            radius=kp_radius,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    combined = cv2.addWeighted(canvas, 0.3, combined, 0.7, 0)

    return combined


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
