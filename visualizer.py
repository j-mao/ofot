from __future__ import annotations

import functools
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from PIL import Image


@functools.lru_cache(maxsize=None)
def initialize_visualizer() -> tuple[np.ndarray, int]:
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.concatenate([
        np.array([[1, 0, 0]]) + np.arange(ry)[:, None] * np.array([[0, +1./ry, 0]]),
        np.array([[1, 1, 0]]) + np.arange(yg)[:, None] * np.array([[-1./yg, 0, 0]]),
        np.array([[0, 1, 0]]) + np.arange(gc)[:, None] * np.array([[0, 0, +1./gc]]),
        np.array([[0, 1, 1]]) + np.arange(cb)[:, None] * np.array([[0, -1./cb, 0]]),
        np.array([[0, 0, 1]]) + np.arange(bm)[:, None] * np.array([[+1./bm, 0, 0]]),
        np.array([[1, 0, 1]]) + np.arange(mr)[:, None] * np.array([[0, 0, -1./mr]]),
    ])
    return colorwheel, ncols


def visualize(flow: np.ndarray, *, save: Optional[str]) -> None:
    colorwheel, ncols = initialize_visualizer()

    rad = np.linalg.norm(flow, axis=-1)
    rad /= rad.max()

    a = np.arctan2(-flow[..., 1], -flow[..., 0]) / np.pi
    fk = (a + 1.0) / 2 * (ncols - 1)
    k0 = fk.astype(int)
    k1 = (k0 + 1) % ncols
    f = (fk - k0)[..., None]

    col = (1 - f) * colorwheel[k0] + f * colorwheel[k1]
    col = 1 - rad[..., None] * (1 - col)  # Increase saturation with radius.

    plt.imshow(col)
    plt.show()
    if save is not None:
        img = Image.fromarray((col * 255).astype(np.uint8), "RGB")
        img.save(save)
