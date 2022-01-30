"""
Evaluate a sizer's score of a given video.

This can be used to validate if the sizer truly prefers an incorrect shape or
if something went wrong in the (local) search process.
"""

import argparse

import numpy as np
from video_autosize.heuristic import JPEGSizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizer", default="jpeg", type=str, help='options: "jpeg"')
    parser.add_argument("video_npz_path", default=None, type=str)
    args = parser.parse_args()

    sizer = {
        "jpeg": JPEGSizer(),
    }[args.sizer]

    loaded = np.load(args.video_npz_path)
    video = loaded["arr_0"]
    print(sizer.video_score(video))


if __name__ == "__main__":
    main()
