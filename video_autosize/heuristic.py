import io
from email.quoprimime import header_check

import numpy as np
from PIL import Image

from .base import ScoreSizer


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
