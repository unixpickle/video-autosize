"""
Evaluate a sizer on a single video or a dataset of videos.

Pass the sizer using the --sizer argument. For example, `--sizer deltas`.

By default, this will download a dataset and run over the entire test set.
To run on specific file(s), use `--video_npz_path PATH1 PATH2 ...`.
"""

import argparse
import glob
import os

import numpy as np
from video_autosize.base import truncated_reshape
from video_autosize.data import download_dataset
from video_autosize.heuristic import named_sizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizer",
        default="jpeg",
        type=str,
        help=f"options: {','.join(named_sizers().keys())}",
    )
    parser.add_argument("--use_frames", default=3.5, type=float)
    parser.add_argument("--video_npz_path", default=None, type=str, nargs="+")
    parser.add_argument("--output_npz_path", default=None, type=str)
    args = parser.parse_args()

    sizer = named_sizers()[args.sizer]

    if args.video_npz_path is None:
        print("downloading dataset and using test split...")
        local_path = download_dataset()
        video_paths = glob.glob(os.path.join(local_path, "test", "*.npz"))
    else:
        video_paths = args.video_npz_path

    correct_counts = dict(width=0, height=0, both=0)
    total_count = 0

    for video_path in video_paths:
        loaded = np.load(video_path)
        video = loaded["arr_0"]

        frame_size = np.prod(video.shape[1:-1])
        num_pixels = int(args.use_frames * frame_size)
        flat = video.reshape([-1, 3])[:num_pixels]

        print(f"working on {video_path}")
        print(
            f"  - trying {len(flat)} pixels of video with shape {video.shape}"
            f" ({len(flat)/frame_size:.1f} frames)"
        )
        pred_w, pred_h = sizer.predict_resolution(flat, verbose_indent="  - ")
        print(f"  => prediction {pred_w}x{pred_h}")

        w_correct = pred_w == video.shape[2]
        h_correct = pred_h == video.shape[1]
        if w_correct:
            correct_counts["width"] += 1
        if h_correct:
            correct_counts["height"] += 1
        if w_correct and h_correct:
            correct_counts["both"] += 1
        total_count += 1

        if args.output_npz_path is not None:
            np.savez(
                args.output_npz_path,
                truncated_reshape(video, pred_w, pred_h),
                fps=loaded["fps"],
            )

    print("*** final correctness per dimension:")
    for k, v in correct_counts.items():
        print(f"  - {k}: {v}/{total_count} ({100*v/total_count:.02f}%)")


if __name__ == "__main__":
    main()
