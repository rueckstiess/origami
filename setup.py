import re

from setuptools import find_packages, setup

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    # Remove both image and arxiv link sections
    long_description = re.sub(
        r'<p align="center">(?:\s*<img[^>]*>|\s*\|[^|]*\|)\s*</p>\s*\n?', "", long_description, flags=re.MULTILINE
    )

    # Remove the Disclaimer section (from ## Disclaimer to the next ##)
    long_description = re.sub(r"## Disclaimer.*?(?=## \w+)", "", long_description, flags=re.DOTALL)

setup(
    name="origami-ml",
    author="Thomas Rueckstiess",
    author_email="me@tomr.au",
    description="An ML classifier model to make predictions from semi-structured data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rueckstiess/origami",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "origami = origami.cli:main",
        ],
    },
    install_requires=[
        "click>=8.1.7",
        "click-option-group>=0.5.6",
        "guildai>=0.9.0",
        "lightgbm>=4.5.0",
        "matplotlib>=3.9.2",
        "mdbrtools>=0.1.1",
        "numpy>=1.26.4",
        "omegaconf>=2.3.0",
        "openml>=0.15.1",
        "pandas>=2.2.3",
        "pymongo>=4.8.0",
        "python-dotenv>=1.0.1",
        "scikit_learn>=1.5.2",
        "torch>=2.4.1",
        "tqdm>=4.66.4",
        "xgboost>=2.1.3",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.1.1",
            "jupyter_contrib_nbextensions>=0.7.0",
            "pytest>=8.3.3",
            "ruff>=0.9.3",
        ],
    },
)
