import io

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
        compressed_bytes = 0
        total_bytes = int(np.prod(vid.shape))
        for frame in vid:
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="jpeg")
            buf.seek(0)
            compressed_bytes += len(buf.read())
            del buf
        return total_bytes / compressed_bytes
