from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AutoSizer(ABC):
    @abstractmethod
    def predict_resolution(self, pixels: np.ndarray) -> Tuple[int, int]:
        """
        Predict the resolution of a stream of pixels.

        :param pixels: an [N x 3] array of uint8 pixels.
        :return: a tuple (width, height) of predicted dimensions.
        """


class ScoreSizer(AutoSizer):
    def __init__(self, min_height: int = 64, max_width: int = 2048):
        self.min_height = min_height
        self.max_width = max_width

    def predict_resolution(self, pixels: np.ndarray) -> Tuple[int, int]:
        assert len(pixels) > self.min_height
        best_width = None
        best_score = 0.0
        for width in range(1, self.max_width + 1):
            height = len(pixels) // width
            if height < self.min_height:
                break
            reshaped = pixels[: width * height].reshape([height, width, 3])
            score = self.image_score(reshaped)
            if best_width is None or score > best_score:
                best_width = width
                best_score = score

        best_height = None
        best_score = 0.0
        max_height = len(pixels) // best_width
        for height in range(self.min_height, max_height):
            num_frames = len(pixels) // (height * best_width)
            reshaped = pixels[: best_width * height * num_frames].reshape(
                [num_frames, height, best_width, 3]
            )
            score = self.video_score(reshaped)
            if best_height is None or score > best_score:
                best_height = height
                best_score = score

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
