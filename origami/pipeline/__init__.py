"""ORIGAMI Pipeline - unified training and inference API.

This module provides a high-level API for training ORIGAMI models on JSON data
and performing inference with automatic preprocessing and inverse transforms.

Example:
    ```python
    from origami.pipeline import OrigamiPipeline, OrigamiConfig

    # Simple training with defaults
    pipeline = OrigamiPipeline()
    pipeline.fit(train_data, epochs=20)
    pipeline.save("model.pt")

    # Load and inference
    pipeline = OrigamiPipeline.load("model.pt")
    prediction = pipeline.predict({"a": 3.5}, target_key="b")
    samples = pipeline.generate(num_samples=10)
    ```
"""

from origami.config import OrigamiConfig

from .pipeline import OrigamiPipeline

__all__ = ["OrigamiPipeline", "OrigamiConfig"]
