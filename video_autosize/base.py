from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class AutoSizer(ABC):
    @abstractmethod
    def predict_resolution(
        self, pixels: np.ndarray, verbose_indent: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Predict the resolution of a stream of pixels.

        :param pixels: an [N x 3] array of uint8 pixels.
        :param verbose_indent: if specified, print intermediate progress where
                               each line has this prefix.
        :return: a tuple (width, height) of predicted dimensions.
        """


class ScoreSizer(AutoSizer):
    def __init__(
        self,
        min_width: int = 64,
        min_height: int = 64,
        max_width: int = 2048,
        max_height: int = 2048,
        max_pixels: int = 2048 ** 2,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height
        self.max_pixels = max_pixels

    def predict_resolution(
        self, pixels: np.ndarray, verbose_indent: Optional[str] = None
    ) -> Tuple[int, int]:
        assert len(pixels) > self.min_height
        best_width = None
        best_score = 0.0
        if verbose_indent is not None:
            print(f"{verbose_indent}inferring width...")
        for width in range(self.min_width, self.max_width + 1):
            height = min(len(pixels) // width, self.max_pixels // width)
            if height < self.min_height:
                break
            height = min(height, self.max_height)
            reshaped = pixels[: width * height].reshape([height, width, 3])
            score = self.image_score(reshaped)
            if best_width is None or score > best_score:
                best_width = width
                best_score = score

        if verbose_indent is not None:
            print(f"{verbose_indent}inferred width: {best_width} (score {best_score})")

        best_height = None
        best_score = 0.0
        max_height = min(
            len(pixels) // best_width, self.max_pixels // best_width, self.max_height
        )
        for height in range(self.min_height, max_height):
            num_frames = len(pixels) // (height * best_width)
            reshaped = pixels[: best_width * height * num_frames].reshape(
                [num_frames, height, best_width, 3]
            )
            score = self.video_score(reshaped)
            if best_height is None or score > best_score:
                best_height = height
                best_score = score

        if verbose_indent is not None:
            print(
                f"{verbose_indent}inferred height: {best_height} (score {best_score})"
            )

        return best_width, best_height

    def image_score(self, img: np.ndarray) -> float:
        """
        Similar to video_score(), but score a single [H x W x 3] image.

        :param img: a [H x W x 3] video array.
        :return: a score estimating the correctness of the resolution.
        """
        return self.video_score(img[None])

    @abstractmethod
    def video_score(self, vid: np.ndarray) -> float:
        """
        Produce a score measuring how likely it is that the resolution of a
        video is correct. The score should be relative to the number of pixels,
        so that slightly changing the number of pixels while searching
        different resolutions doesn't change relative scores.

        :param vid: a [T x H x W x 3] video array.
        :return: a score estimating the correctness of the resolution.
        """


def truncated_reshape(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    pixels = pixels.reshape([-1, 3])
    num_frames = len(pixels) // (height * width)
    return pixels[: width * height * num_frames].reshape([num_frames, height, width, 3])
