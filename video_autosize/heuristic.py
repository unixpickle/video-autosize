import io
from typing import Dict

import numpy as np
from PIL import Image

from .base import ScoreSizer


def named_sizers() -> Dict[str, ScoreSizer]:
    return {
        "jpeg": JPEGSizer(),
        "delta": DeltaSizer(),
    }


class JPEGSizer(ScoreSizer):
    """
    Use JPEG compression to evaluate when images are "natural looking".
    In particular, natural images compress better than distorted ones.
    """

    def __init__(self, quality: int = 85, **kwargs):
        super().__init__(**kwargs)
        self.quality = quality

    def video_score(self, vid: np.ndarray) -> float:
        header_len = self._measure_image(vid[0, :1, :1])
        compressed_bytes = 0
        total_bytes = int(np.prod(vid.shape))
        for frame in vid:
            compressed_bytes += self._measure_image(frame) - header_len
        return total_bytes / compressed_bytes

    def _measure_image(self, arr: np.ndarray) -> int:
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="jpeg", quality=self.quality)
        buf.seek(0)
        return len(buf.read())


class DeltaSizer(ScoreSizer):
    def image_score(self, img: np.ndarray) -> float:
        return -self._neighbor_diffs(img, 2)

    def video_score(self, vid: np.ndarray) -> float:
        return -self._neighbor_diffs(vid, 3)

    def _neighbor_diffs(self, arr: np.ndarray, num_dims: int) -> float:
        diff_count = np.zeros_like(arr)
        diff_sum = np.zeros_like(arr)
        for dim in range(num_dims):
            prefix = (
                ((slice(None),) * dim)
                + (slice(None, -1),)
                + (
                    (
                        slice(
                            None,
                        ),
                    )
                    * (num_dims - (dim + 1))
                )
            )
            suffix = (
                ((slice(None),) * dim)
                + (slice(1, None),)
                + (
                    (
                        slice(
                            None,
                        ),
                    )
                    * (num_dims - (dim + 1))
                )
            )
            diff_count[prefix] += 1
            diff_count[suffix] += 1
            diff = (arr[prefix] - arr[suffix]) ** 2
            diff_sum[prefix] += diff
            diff_sum[suffix] += diff
        return np.sum(diff_sum) / np.sum(diff_count)
