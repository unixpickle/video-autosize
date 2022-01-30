import argparse
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
    parser.add_argument("--video_npz_path", default=None, type=str)
    parser.add_argument("--output_npz_path", default=None, type=str)
    args = parser.parse_args()

    sizer = named_sizers()[args.sizer]

    if args.video_npz_path is None:
        print("downloading dataset and using default video...")
        local_path = download_dataset()
        video_path = os.path.join(local_path, "test", "---v8pgm1eQ.npz")
    else:
        video_path = args.video_npz_path

    loaded = np.load(video_path)
    video = loaded["arr_0"]

    frame_size = np.prod(video.shape[1:-1])
    num_pixels = int(args.use_frames * frame_size)
    flat = video.reshape([-1, 3])[:num_pixels]

    print(
        f"trying {len(flat)} pixels of video with shape {video.shape}"
        f" ({len(flat)/frame_size:.1f} frames)"
    )
    pred_w, pred_h = sizer.predict_resolution(flat, verbose_indent="  - ")
    print(f"  => prediction {pred_w}x{pred_h}")

    if args.output_npz_path is not None:
        np.savez(
            args.output_npz_path,
            truncated_reshape(video, pred_w, pred_h),
            fps=loaded["fps"],
        )


if __name__ == "__main__":
    main()
