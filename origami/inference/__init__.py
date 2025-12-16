"""ORIGAMI inference utilities.

Provides inference modes for trained ORIGAMI models:
- Embedder: Extract document embeddings
- Predictor: Predict values for target keys
- Generator: Generate complete JSON objects
"""

from .embedder import OrigamiEmbedder
from .generator import OrigamiGenerator
from .predictor import OrigamiPredictor

__all__ = [
    "OrigamiEmbedder",
    "OrigamiGenerator",
    "OrigamiPredictor",
]
