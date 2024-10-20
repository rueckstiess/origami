from distutils.core import setup

setup(
    name="storm_ml", 
    version="0.1.0", 
    packages=["storm_ml"],
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts': [
            'storm = storm_ml.cli:main',
        ],
    },
)
