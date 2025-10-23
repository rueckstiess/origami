import pytest

from origami.inference import Sampler
from origami.preprocessing import StreamEncoder
from origami.utils import Symbol


@pytest.fixture
def mock_model():
    """Create a mock model for testing (no training required)."""
    from unittest.mock import Mock

    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.model_config.guardrails = False
    model.model_config.block_size = 50
    model.device = "cpu"

    return model


@pytest.fixture
def encoder():
    """Create a real StreamEncoder."""
    encoder = StreamEncoder(predefined=Symbol)
    encoder.freeze()
    return encoder


class TestSamplerConstructor:
    """Test Sampler constructor and initialization."""

    def test_constructor_with_schema(self, mock_model, encoder):
        """Test that constructor properly initializes with schema."""
        from mdbrtools.schema import Schema

        schema = Schema()

        sampler = Sampler(mock_model, encoder, schema, temperature=1.0)

        assert sampler.model is mock_model
        assert sampler.encoder is encoder
        assert sampler.schema is schema
        assert sampler.temperature == 1.0
        assert sampler.vpda is not None

    def test_constructor_without_schema(self, mock_model, encoder):
        """Test that constructor works without schema (no guardrails)."""
        sampler = Sampler(mock_model, encoder, schema=None, temperature=1.0)

        assert sampler.model is mock_model
        assert sampler.encoder is encoder
        assert sampler.schema is None
        assert sampler.vpda is None

    def test_constructor_custom_temperature(self, mock_model, encoder):
        """Test constructor with custom temperature."""
        from mdbrtools.schema import Schema

        schema = Schema()
        sampler = Sampler(mock_model, encoder, schema, temperature=0.5)

        assert sampler.temperature == 0.5

    def test_constructor_default_temperature(self, mock_model, encoder):
        """Test that default temperature is 1.0."""
        sampler = Sampler(mock_model, encoder)

        assert sampler.temperature == 1.0


# NOTE: Integration tests for sample() method require a trained model
# and are difficult to test in unit tests. The Sampler should be tested
# manually with a trained model (see notebooks/example_origami_mc_ce.ipynb).
#
# The core logic (constructor, temperature handling, etc.) is tested above.
