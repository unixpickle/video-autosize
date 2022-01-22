import random
from typing import List, Tuple

import numpy as np
from video_autosize.base import AutoSizer
from video_autosize.data import iterate_videos
from video_autosize.heuristic import JPEGSizer


def main():
    sizers = {
        "jpeg": JPEGSizer(),
    }
    for name, sizer in sizers.items():
        print(f"evaluating {name}...")
        frac, incorrect = evaluate_sizer(sizer)
        print(f" => correctness fraction: {frac}")


def evaluate_sizer(
    s: AutoSizer,
) -> Tuple[float, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    num_correct = 0
    num_total = 0
    incorrect_sizes = []
    for video in iterate_videos(split="test", loop=False, max_frames=5):
        frame_size = int(np.prod(video.shape[1:-1]))
        total_size = int(np.prod(video.shape[:-1]))
        flat = video.reshape([-1, 3])[: random.randrange(frame_size, total_size + 1)]
        print(
            f" trying {len(flat)} pixels of video with shape {video.shape}"
            f" ({len(flat)/frame_size:.1f} frames)"
        )
        pred_w, pred_h = s.predict_resolution(flat, verbose_indent="   - ")
        print(f"   => prediction {pred_w}x{pred_h}")
        _, real_h, real_w, _ = video.shape
        if (pred_w, pred_h) != (real_w, real_h):
            incorrect_sizes.append(((pred_w, pred_h), (real_w, real_h)))
        else:
            num_correct += 1
        num_total += 1
    return num_correct / num_total, incorrect_sizes


if __name__ == "__main__":
    main()
