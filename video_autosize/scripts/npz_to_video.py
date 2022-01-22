import argparse
import os
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_npz", type=str)
    parser.add_argument("out_mp4", type=str)
    args = parser.parse_args()

    loaded = np.load(args.in_npz)
    frames = loaded["arr_0"]
    fps = loaded["fps"]
    export_video(args.out_mp4, frames.shape[2], frames.shape[1], fps, frames)


def export_video(path, width, height, fps, frames):
    # https://github.com/unixpickle/anyrl-py/blob/c70758463bd4cd558f809722c3e1aa1c0c54ff1c/anyrl/utils/ffmpeg.py#L12
    video_reader, video_writer = os.pipe()

    video_format = [
        "-r",
        str(fps),
        "-s",
        "%dx%d" % (width, height),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
    ]
    video_params = video_format + [
        "-probesize",
        "32",
        "-thread_queue_size",
        "10000",
        "-i",
        "pipe:%i" % video_reader,
    ]
    output_params = [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-f",
        "mp4",
        "-pix_fmt",
        "yuv420p",
        path,
    ]
    ffmpeg_proc = subprocess.Popen(
        ["ffmpeg", "-y", *video_params, *output_params],
        pass_fds=(video_reader,),
        stdin=subprocess.DEVNULL,
    )
    try:
        for img in frames:
            assert img.shape == (height, width, 3)
            os.write(video_writer, bytes(img))
    finally:
        os.close(video_writer)
        ffmpeg_proc.wait()


if __name__ == "__main__":
    main()
