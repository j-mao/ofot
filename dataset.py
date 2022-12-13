from __future__ import annotations

import functools
import itertools
import os
from typing import Generator

import numpy as np
from PIL import Image


dataset_name = "training"  # training or test
dataset_quality = "final"  # final or clean


@functools.lru_cache(maxsize=None)
def get_image(task: str, frame: int) -> Image.Image:
    return Image.open(os.path.join("data", dataset_name, dataset_quality, task, f"frame_{frame:04}.png"))


@functools.lru_cache(maxsize=None)
def get_flow(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        assert np.fromfile(f, dtype=np.float32, count=1) == 202021.25  # TAG
        w, h = np.fromfile(f, dtype=np.int32, count=2)
        nbands = 2
        return np.fromfile(f, dtype=np.float32).reshape((h, w, nbands))


def get_ground_truth(task: str, frame: int) -> np.ndarray:
    try:
        return get_flow(os.path.join("data", dataset_name, "flow", task, f"frame_{frame:04}.flo"))
    except FileNotFoundError:
        return np.zeros((436, 1024, 2), dtype=np.float32)


def get_predicted_flow(task: str, frame: int) -> np.ndarray:
    return get_flow(os.path.join("data", "output", dataset_quality, task, f"frame_{frame:04}.flo"))


def write_flow(flow: np.ndarray, task: str, frame: int, cmt: str) -> None:
    with open(os.path.join("data", "output", dataset_quality, task, f"frame_{frame:04}_{cmt}.flo"), "wb") as f:
        np.float32(202021.25).tofile(f)  # TAG
        np.array(flow.shape[1::-1], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def list_tasks() -> Generator[tuple[str, int], None, None]:
    for i in itertools.count(1):
        for task in os.listdir(os.path.join("data", dataset_name, dataset_quality)):
            if os.path.exists(os.path.join("data", dataset_name, dataset_quality, task, f"frame_{i+1:04}.png")):
                yield task, i
            else:
                continue
