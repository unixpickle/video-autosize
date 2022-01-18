"""
Download a handful of videos from a Kinetics CSV dataset.

To download the metadata, see https://deepmind.com/research/open-source/kinetics.
"""

import argparse
import csv
import os
import subprocess
from typing import Optional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(args.csv_path, "rt") as f:
        reader = csv.DictReader(f)
        for record in reader:
            youtube_id = record["youtube_id"]
            out_path = os.path.join(args.output_dir, f"{youtube_id}.mp4")
            if os.path.exists(out_path):
                continue
            video_url = get_video_url(record["youtube_id"])
            if video_url is None:
                continue
            download_video(video_url, record["time_start"], out_path)


def get_video_url(youtube_id: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            [
                "youtube-dl",
                "--rm-cache-dir",
                "--youtube-skip-dash-manifest",
                "-g",
                f"https://www.youtube.com/watch?v={youtube_id}",
            ]
        )
        out = str(out, "utf-8")
        lines = out.splitlines()
        return lines[0].strip()
    except subprocess.CalledProcessError:
        return None


def download_video(video_url: str, start: str, out_path: str):
    try:
        subprocess.check_output(
            [
                "ffmpeg",
                "-ss",
                start,
                "-i",
                video_url,
                "-ss",
                "5",
                "-t",
                "5",
                out_path,
            ]
        )
    except subprocess.CalledProcessError:
        return None


if __name__ == "__main__":
    main()
