"""ORIGAMI: Object RepresentatIon via Generative Autoregressive ModellIng.

A transformer-based architecture for supervised learning on semi-structured JSON data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("origami-ml")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0+unknown"

# Re-export key classes for convenience
from origami.config import DataConfig, InferenceConfig, ModelConfig, OrigamiConfig, TrainingConfig
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
    "InferenceConfig",
    # Model (advanced usage)
    "OrigamiModel",
    "OrigamiOutput",
    # Tokenizer
    "JSONTokenizer",
    "EncodedBatch",
    "Vocabulary",
]
