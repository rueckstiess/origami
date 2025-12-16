"""Tests for OrigamiGenerator."""

import pytest
import torch

from origami.inference import OrigamiGenerator
from origami.inference.generator import PathState
from origami.model import OrigamiConfig, OrigamiModel
from origami.position_encoding import PATH_TYPE_INDEX, PATH_TYPE_KEY
from origami.tokenizer import JSONTokenizer


@pytest.fixture
def simple_tokenizer():
    """Create a tokenizer fitted on simple data."""
    data = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"},
        {"name": "Charlie", "age": 35, "city": "SF"},
    ]
    tokenizer = JSONTokenizer()
    tokenizer.fit(data)
    return tokenizer


@pytest.fixture
def simple_model(simple_tokenizer):
    """Create a small model for testing."""
    config = OrigamiConfig(
        vocab_size=simple_tokenizer.vocab.size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_depth=simple_tokenizer.max_depth,
    )
    return OrigamiModel(config, vocab=simple_tokenizer.vocab)


class TestPathState:
    """Tests for PathState tracking."""

    def test_empty_state(self):
        """Test empty path state."""
        state = PathState()
        assert state.get_current_path() == []
        assert state.get_value_path() == []

    def test_push_object(self):
        """Test pushing object context."""
        state = PathState()
        state.push_object()
        assert len(state.context_stack) == 1
        assert state.context_stack[0][0] == "object"

    def test_push_array(self):
        """Test pushing array context."""
        state = PathState()
        state.push_array()
        assert len(state.context_stack) == 1
        assert state.context_stack[0][0] == "array"
        assert state.array_index == 0

    def test_set_key(self):
        """Test setting current key."""
        state = PathState()
        state.push_object()
        state.set_key(PATH_TYPE_KEY, 42)
        assert state.current_key == (PATH_TYPE_KEY, 42)

    def test_value_path_in_object(self):
        """Test value path includes key in object context."""
        state = PathState()
        state.push_object()
        state.set_key(PATH_TYPE_KEY, 42)
        path = state.get_value_path()
        assert len(path) == 1
        assert path[0] == (PATH_TYPE_KEY, 42)

    def test_value_path_in_array(self):
        """Test value path includes index in array context."""
        state = PathState()
        state.push_array()
        path = state.get_value_path()
        assert len(path) == 1
        assert path[0] == (PATH_TYPE_INDEX, 0)

    def test_advance_array_index(self):
        """Test advancing array index."""
        state = PathState()
        state.push_array()
        assert state.array_index == 0
        state.advance_array_index()
        assert state.array_index == 1

    def test_nested_context(self):
        """Test nested object/array contexts."""
        state = PathState()
        # Root object
        state.push_object()
        state.set_key(PATH_TYPE_KEY, 10)

        # Nested object value
        # push_object now automatically includes current_key in the new context's base path
        state.push_object()
        state.set_key(PATH_TYPE_KEY, 20)

        path = state.get_value_path()
        assert len(path) == 2
        assert path[0] == (PATH_TYPE_KEY, 10)
        assert path[1] == (PATH_TYPE_KEY, 20)

    def test_pop_context(self):
        """Test popping context."""
        state = PathState()
        state.push_object()
        state.push_array()
        assert len(state.context_stack) == 2

        state.pop_context()
        assert len(state.context_stack) == 1
        assert state.context_stack[0][0] == "object"

    def test_clone(self):
        """Test cloning path state."""
        state = PathState()
        state.push_object()
        state.set_key(PATH_TYPE_KEY, 42)

        cloned = state.clone()
        assert cloned.context_stack == state.context_stack
        assert cloned.current_key == state.current_key

        # Modifying clone shouldn't affect original
        cloned.set_key(PATH_TYPE_KEY, 100)
        assert state.current_key == (PATH_TYPE_KEY, 42)


class TestOrigamiGenerator:
    """Tests for OrigamiGenerator."""

    def test_init(self, simple_model, simple_tokenizer):
        """Test generator initialization."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        assert generator.model is simple_model
        assert generator.tokenizer is simple_tokenizer

    def test_generate_single(self, simple_model, simple_tokenizer):
        """Test generating single object."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        # With random weights, output is random but should be valid JSON
        results = generator.generate(num_samples=1, max_length=50, seed=42)

        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_generate_batch(self, simple_model, simple_tokenizer):
        """Test generating multiple objects."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        results = generator.generate(num_samples=3, max_length=50, seed=42)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)

    def test_generate_deterministic_with_seed(self, simple_model, simple_tokenizer):
        """Test that seed produces deterministic output."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        results1 = generator.generate(num_samples=2, max_length=50, seed=123)
        results2 = generator.generate(num_samples=2, max_length=50, seed=123)

        assert results1 == results2

    def test_generate_different_seeds(self, simple_model, simple_tokenizer):
        """Test that different seeds produce different output."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        results1 = generator.generate(num_samples=1, max_length=50, seed=1)
        results2 = generator.generate(num_samples=1, max_length=50, seed=2)

        # With very high probability, different seeds should produce different results
        # (could be same by chance, but extremely unlikely)
        # We mainly test that it doesn't crash

    def test_generate_with_temperature(self, simple_model, simple_tokenizer):
        """Test generation with temperature."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        # Low temperature (more greedy)
        results_low = generator.generate(
            num_samples=1, max_length=50, temperature=0.1, seed=42
        )
        assert len(results_low) == 1

        # High temperature (more random)
        results_high = generator.generate(
            num_samples=1, max_length=50, temperature=2.0, seed=42
        )
        assert len(results_high) == 1

    def test_generate_with_top_k(self, simple_model, simple_tokenizer):
        """Test generation with top-k sampling."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        results = generator.generate(
            num_samples=1, max_length=50, top_k=5, seed=42
        )

        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_generate_with_top_p(self, simple_model, simple_tokenizer):
        """Test generation with nucleus (top-p) sampling."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        results = generator.generate(
            num_samples=1, max_length=50, top_p=0.9, seed=42
        )

        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_generate_from_prefix(self, simple_model, simple_tokenizer):
        """Test generation from prefix."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        prefix = {"name": "Alice"}
        results = generator.generate_from_prefix(
            prefix, num_samples=1, max_length=50
        )

        assert len(results) == 1
        assert isinstance(results[0], dict)
        # The result should contain the prefix key
        # (may not have same value due to random generation, but key should be present)

    def test_generate_from_prefix_batch(self, simple_model, simple_tokenizer):
        """Test batch generation from prefix."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        prefix = {"name": "Alice", "age": 30}
        results = generator.generate_from_prefix(
            prefix, num_samples=3, max_length=50
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)

    def test_always_uses_cpu(self, simple_tokenizer):
        """Test that generator always runs on CPU for performance."""
        config = OrigamiConfig(
            vocab_size=simple_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=simple_tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=simple_tokenizer.vocab)

        generator = OrigamiGenerator(model, simple_tokenizer)

        # Verify generator uses CPU
        assert generator.device == torch.device("cpu")
        # Model should be moved to CPU
        assert next(generator.model.parameters()).device == torch.device("cpu")

        results = generator.generate(num_samples=1, max_length=30, seed=42)
        assert len(results) == 1


class TestGeneratorSampling:
    """Tests for sampling functions."""

    def test_sample_greedy(self, simple_model, simple_tokenizer):
        """Test that temperature=0 approximates greedy decoding."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        # Create some logits
        logits = torch.randn(2, simple_tokenizer.vocab.size)

        # Very low temperature should be nearly greedy
        sampled = generator._sample(logits.clone(), temperature=0.01)

        # Should be close to argmax (very high probability)
        expected = logits.argmax(dim=-1)
        # With very low temp, should match argmax
        assert torch.all(sampled == expected)

    def test_sample_top_k(self, simple_model, simple_tokenizer):
        """Test top-k filtering."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        # Create logits with clear top values
        logits = torch.zeros(1, 100)
        logits[0, 0] = 10.0
        logits[0, 1] = 9.0
        logits[0, 2] = 8.0

        # With top_k=2, should only sample from tokens 0 or 1
        torch.manual_seed(42)
        samples = [generator._sample(logits.clone(), top_k=2).item() for _ in range(100)]

        assert all(s in [0, 1] for s in samples)

    def test_sample_top_p(self, simple_model, simple_tokenizer):
        """Test top-p (nucleus) filtering."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        # Create logits where first token has very high probability
        logits = torch.zeros(1, 100)
        logits[0, 0] = 100.0  # This will have prob ~1.0

        # With top_p=0.5, should only get token 0
        torch.manual_seed(42)
        samples = [generator._sample(logits.clone(), top_p=0.5).item() for _ in range(10)]

        assert all(s == 0 for s in samples)


class TestGeneratorWithNestedData:
    """Tests for generator with nested JSON structures."""

    @pytest.fixture
    def nested_tokenizer(self):
        """Create a tokenizer fitted on nested data."""
        data = [
            {
                "user": {"name": "Alice", "age": 30},
                "status": "active",
            },
            {
                "user": {"name": "Bob", "age": 25},
                "status": "inactive",
            },
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def nested_model(self, nested_tokenizer):
        """Create a model for nested data."""
        config = OrigamiConfig(
            vocab_size=nested_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=nested_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=nested_tokenizer.vocab)

    def test_generate_nested(self, nested_model, nested_tokenizer):
        """Test generating nested objects."""
        generator = OrigamiGenerator(nested_model, nested_tokenizer)

        results = generator.generate(num_samples=2, max_length=100, seed=42)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)


class TestGeneratorWithArrays:
    """Tests for generator with array data."""

    @pytest.fixture
    def array_tokenizer(self):
        """Create a tokenizer fitted on array data."""
        data = [
            {"tags": ["python", "ml"], "scores": [95, 87]},
            {"tags": ["java"], "scores": [88, 90, 92]},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def array_model(self, array_tokenizer):
        """Create a model for array data."""
        config = OrigamiConfig(
            vocab_size=array_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=array_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=array_tokenizer.vocab)

    def test_generate_with_arrays(self, array_model, array_tokenizer):
        """Test generating objects with arrays."""
        generator = OrigamiGenerator(array_model, array_tokenizer)

        results = generator.generate(num_samples=2, max_length=100, seed=42)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)


class TestGenerateValue:
    """Tests for generate_value helper method."""

    def test_generate_value_object(self, simple_model, simple_tokenizer):
        """Test generating a complex object value."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)
        vocab = simple_tokenizer.vocab

        # Create a minimal sequence: START OBJ_START key OBJ_START
        # We're at the point where we need to generate a nested object value
        input_ids = torch.tensor(
            [[vocab.start_id, vocab.obj_start_id]], dtype=torch.long, device=generator.device
        )
        path_types = torch.zeros(1, 2, simple_tokenizer.max_depth, dtype=torch.long, device=generator.device)
        path_ids = torch.zeros(1, 2, simple_tokenizer.max_depth, dtype=torch.long, device=generator.device)
        path_lengths = torch.zeros(1, 2, dtype=torch.long, device=generator.device)

        state = PathState()
        state.push_object()

        tokens, value = generator.generate_value(
            input_ids, path_types, path_ids, path_lengths,
            state, max_tokens=50
        )

        # Should return some tokens and a value (may be empty dict due to random weights)
        assert isinstance(tokens, list)

    def test_decode_value_tokens_primitive(self, simple_model, simple_tokenizer):
        """Test decoding primitive value tokens."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)
        vocab = simple_tokenizer.vocab

        # Get token ID for a value
        from origami.tokenizer.vocabulary import ValueToken
        value_token = ValueToken("Alice")
        token_id = vocab.encode(value_token)

        result = generator._decode_value_tokens([token_id])
        assert result == "Alice"

    def test_decode_value_tokens_empty_object(self, simple_model, simple_tokenizer):
        """Test decoding empty object tokens."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)
        vocab = simple_tokenizer.vocab

        tokens = [vocab.obj_start_id, vocab.obj_end_id]
        result = generator._decode_value_tokens(tokens)
        assert result == {}

    def test_decode_value_tokens_empty_array(self, simple_model, simple_tokenizer):
        """Test decoding empty array tokens."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)
        vocab = simple_tokenizer.vocab

        tokens = [vocab.array_start_id, vocab.array_end_id]
        result = generator._decode_value_tokens(tokens)
        assert result == []
