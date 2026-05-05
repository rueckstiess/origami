"""ORIGAMI inference utilities.

Provides inference modes for trained ORIGAMI models:
- Embedder: Extract document embeddings
- Predictor: Predict values for target keys
- Generator: Generate complete JSON objects
- Evaluator: Unified evaluation for loss and accuracy metrics
"""

from .embedder import OrigamiEmbedder
from .evaluator import OrigamiEvaluator, evaluate
from .generator import OrigamiGenerator
from .predictor import OrigamiPredictor
from .utils import GenerationError, find_target_positions

__all__ = [
    "GenerationError",
    "OrigamiEmbedder",
    "OrigamiEvaluator",
    "OrigamiGenerator",
    "OrigamiPredictor",
    "evaluate",
    "find_target_positions",
]
