"""ORIGAMI: Object RepresentatIon via Generative Autoregressive ModellIng.

A transformer-based architecture for supervised learning on semi-structured JSON data.
"""

__version__ = "0.1.0"

# Re-export key classes for convenience
from origami.model import OrigamiConfig, OrigamiModel, OrigamiOutput
from origami.tokenizer import EncodedBatch, JSONTokenizer, Vocabulary

__all__ = [
    # Version
    "__version__",
    # Model
    "OrigamiConfig",
    "OrigamiModel",
    "OrigamiOutput",
    # Tokenizer
    "JSONTokenizer",
    "EncodedBatch",
    "Vocabulary",
]
