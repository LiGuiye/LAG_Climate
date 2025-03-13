"""LAG_Climate setup file."""
from setuptools import setup, find_packages

setup(
    name="lag",
    version="0.0.1",
    description="Generative Adversarial Models for Extreme Geospatial Downscaling",
    author="Guiye Li and Guofeng Cao",
    author_email="Guiye.Li@colorado.edu",
    url="https://github.com/LiGuiye/LAG_Climate",
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        "torch==1.13.1+cu116",
        "torchvision==0.14.1+cu116",
        "torchaudio==0.13.1",
        "numpy==1.26.4",
        "matplotlib",
        "bunch",
        "scipy",
        "lpips"
    ],
    python_requires=">=3.7",
)