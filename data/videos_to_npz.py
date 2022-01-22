"""
Convert a directory of videos into npz files.

The npz files contain the following keys:
 - arr_0: a [T x H x W x 3] uint8 array of frames
 - fps: the (floating point) frame rate of the video
"""

import argparse
import os
import re
import subprocess
from typing import Dict, List, Union

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_directory", type=str)
    parser.add_argument("out_directory", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.out_directory):
        os.mkdir(args.out_directory)

    for in_name in os.listdir(args.in_directory):
        in_path = os.path.join(args.in_directory, in_name)
        try:
            info = video_info(in_path)
            frames = read_frames(in_path, info)
        except Exception as exc:
            print(f"failed to read {in_path}")
            print(exc)
            continue
        out_path = os.path.join(
            args.out_directory, os.path.splitext(in_name)[0] + ".npz"
        )
        np.savez(out_path, frames, fps=info["fps"])


def read_frames(path: str, info: Dict[str, Union[int, float]]) -> np.ndarray:
    """
    Read the frames of a video file as an RGB numpy array.
    """
    video_reader, video_writer = os.pipe()
    try:
        args = [
            "ffmpeg",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:%i" % video_writer,
        ]
        ffmpeg_proc = subprocess.Popen(
            args,
            pass_fds=(video_writer,),
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        os.close(video_writer)
        video_writer = -1
        frame_shape = [info["height"], info["width"], 3]
        frame_size = int(np.prod(frame_shape))
        reader = os.fdopen(video_reader, "rb")
        frames = []
        while True:
            buf = reader.read(frame_size)
            if len(buf) < frame_size:
                break
            frames.append(np.frombuffer(buf, dtype="uint8").reshape(frame_shape))
        ffmpeg_proc.wait()
        return np.stack(frames, axis=0)
    finally:
        os.close(video_reader)
        if video_writer >= 0:
            os.close(video_writer)


def video_info(path: str) -> Dict[str, Union[int, float]]:
    """
    https://github.com/unixpickle/anyrl-py/blob/c70758463bd4cd558f809722c3e1aa1c0c54ff1c/anyrl/utils/ffmpeg.py#L166
    """
    result = {}
    for line in _ffmpeg_output_lines(path):
        if "Video:" not in line:
            continue
        match = re.search(" ([0-9]+)x([0-9]+)(,| )", line)
        if match:
            result["width"] = int(match.group(1))
            result["height"] = int(match.group(2))
        match = re.search(" ([0-9\\.]*) fps,", line)
        if match:
            result["fps"] = float(match.group(1))
    if "width" not in result:
        raise ValueError("no dimensions found in output")
    if "fps" not in result:
        raise ValueError("no fps found in output")
    return result


def _ffmpeg_output_lines(path: str) -> List[str]:
    proc = subprocess.Popen(
        ["ffmpeg", "-i", path], stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    _, output = proc.communicate()
    return str(output, "utf-8").split("\n")


if __name__ == "__main__":
    main()
