from setuptools import setup

setup(
    name="video-autosize",
    packages=["video_autosize"],
    install_requires=[
        "Pillow",
        "requests",
        "tqdm",
    ],
    author="Alex Nichol",
)
