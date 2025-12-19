"""ORIGAMI: Object RepresentatIon via Generative Autoregressive ModellIng.

A transformer-based architecture for supervised learning on semi-structured JSON data.
"""

__version__ = "0.1.0"

# Re-export key classes for convenience
from origami.config import DataConfig, ModelConfig, OrigamiConfig, TrainingConfig
from origami.model import OrigamiModel, OrigamiOutput
from origami.pipeline import OrigamiPipeline
from origami.tokenizer import EncodedBatch, JSONTokenizer, Vocabulary

__all__ = [
    # Version
    "__version__",
    # Pipeline (recommended API)
    "OrigamiPipeline",
    # Configuration
    "OrigamiConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # Model (advanced usage)
    "OrigamiModel",
    "OrigamiOutput",
    # Tokenizer
    "JSONTokenizer",
    "EncodedBatch",
    "Vocabulary",
]
