from distutils.core import setup

setup(
    name="origami-ml",
    version="0.1.0",
    packages=["origami"],
    install_requires=[
        "click",
    ],
    entry_points={
        "console_scripts": [
            "origami = origami.cli:main",
        ],
    },
)
