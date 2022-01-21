import os
import random
import tempfile
import zipfile
from typing import Iterator

import numpy as np
import requests
from tqdm.auto import tqdm

DATASET_URL = "https://data.aqnichol.com/video-autosize.zip"


def iterate_videos(
    split: str = "train", max_frames: int = 5, loop: bool = True, **download_kwargs
) -> Iterator[np.ndarray]:
    local_path = download_dataset(**download_kwargs)
    split_dir = os.path.join(local_path, split)
    while True:
        for name in os.listdir(split_dir):
            if not name.endswith(".npz") or name.startswith("."):
                continue
            video = np.load(os.path.join(split_dir, name))["arr_0"]
            if len(video) > max_frames:
                start = random.randrange(len(video) - max_frames)
                video = video[start : start + max_frames]
            yield video
        if not loop:
            break


def download_dataset(
    url: str = DATASET_URL,
    output_path: str = "./video_data",
    progress: bool = True,
    chunk_size: int = 4096,
) -> str:
    """
    Download a zip file and extract it into a given directory.
    """
    if os.path.exists(output_path):
        return output_path
    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "data.zip")
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        if progress:
            pbar.close()
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(output_path)
    return output_path
