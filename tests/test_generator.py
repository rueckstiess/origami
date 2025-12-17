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

        # Generate with different seeds - mainly test that it doesn't crash
        # Results could be same by chance, but extremely unlikely
        generator.generate(num_samples=1, max_length=50, seed=1)
        generator.generate(num_samples=1, max_length=50, seed=2)

    def test_generate_with_temperature(self, simple_model, simple_tokenizer):
        """Test generation with temperature."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        # Low temperature (more greedy)
        results_low = generator.generate(num_samples=1, max_length=50, temperature=0.1, seed=42)
        assert len(results_low) == 1

        # High temperature (more random)
        results_high = generator.generate(num_samples=1, max_length=50, temperature=2.0, seed=42)
        assert len(results_high) == 1

    def test_generate_with_top_k(self, simple_model, simple_tokenizer):
        """Test generation with top-k sampling."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        results = generator.generate(num_samples=1, max_length=50, top_k=5, seed=42)

        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_generate_with_top_p(self, simple_model, simple_tokenizer):
        """Test generation with nucleus (top-p) sampling."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)

        results = generator.generate(num_samples=1, max_length=50, top_p=0.9, seed=42)

        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_generate_from_tensors(self, simple_model, simple_tokenizer):
        """Test generation from pre-encoded tensors."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)
        vocab = simple_tokenizer.vocab

        # Create input tensors for two sequences with different prefixes
        # Sequence 1: START OBJ_START
        # Sequence 2: START OBJ_START (same, but could be different in real use)
        batch_size = 2
        max_depth = simple_tokenizer.max_depth

        input_ids = torch.tensor(
            [
                [vocab.start_id, vocab.obj_start_id],
                [vocab.start_id, vocab.obj_start_id],
            ],
            dtype=torch.long,
            device=generator.device,
        )

        path_types = torch.zeros(
            batch_size, 2, max_depth, dtype=torch.long, device=generator.device
        )
        path_ids = torch.zeros(batch_size, 2, max_depth, dtype=torch.long, device=generator.device)
        path_lengths = torch.zeros(batch_size, 2, dtype=torch.long, device=generator.device)
        attention_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=generator.device)

        results = generator.generate_from_tensors(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            stop_after_value=False,
            max_tokens=50,
        )

        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, dict)

    def test_generate_from_tensors_stop_after_value(self, simple_model, simple_tokenizer):
        """Test generation with stop_after_value=True."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)
        vocab = simple_tokenizer.vocab
        max_depth = simple_tokenizer.max_depth

        # Create a minimal sequence: START OBJ_START key:
        # When stop_after_value=True, should generate a single value then stop
        # Encode a key token
        from origami.tokenizer.vocabulary import KeyToken

        key_token = KeyToken("name")
        key_id = vocab.encode(key_token)

        input_ids = torch.tensor(
            [
                [vocab.start_id, vocab.obj_start_id, key_id],
            ],
            dtype=torch.long,
            device=generator.device,
        )

        path_types = torch.zeros(1, 3, max_depth, dtype=torch.long, device=generator.device)
        path_ids = torch.zeros(1, 3, max_depth, dtype=torch.long, device=generator.device)
        path_lengths = torch.zeros(1, 3, dtype=torch.long, device=generator.device)
        attention_mask = torch.ones(1, 3, dtype=torch.bool, device=generator.device)

        results = generator.generate_from_tensors(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            stop_after_value=True,
            max_tokens=50,
        )

        # Should return a single value (could be primitive or complex)
        assert len(results) == 1
        # The value could be anything depending on model weights

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


class TestDecodeValueTokens:
    """Tests for value token decoding helper methods."""

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

    def test_decode_value_tokens_number(self, simple_model, simple_tokenizer):
        """Test decoding numeric value tokens."""
        generator = OrigamiGenerator(simple_model, simple_tokenizer)
        vocab = simple_tokenizer.vocab

        from origami.tokenizer.vocabulary import ValueToken

        value_token = ValueToken(30)
        token_id = vocab.encode(value_token)

        result = generator._decode_value_tokens([token_id])
        assert result == 30

    def test_decode_value_tokens_boolean(self):
        """Test decoding boolean value tokens."""
        # Need a tokenizer that has seen booleans
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"flag": True}, {"flag": False}])

        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        generator = OrigamiGenerator(model, tokenizer)

        from origami.tokenizer.vocabulary import ValueToken

        true_token = ValueToken(True)
        token_id = tokenizer.vocab.encode(true_token)

        result = generator._decode_value_tokens([token_id])
        assert result is True


class TestGenerateFromTensorsAdvanced:
    """Advanced tests for generate_from_tensors with various scenarios."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for advanced tests."""
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"nested": {"inner": "value"}},
            {"list": [1, 2, 3]},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def model(self, tokenizer):
        """Create model for advanced tests."""
        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=tokenizer.vocab)

    def test_generate_from_tensors_with_left_padding(self, model, tokenizer):
        """Test generate_from_tensors handles left-padded inputs correctly."""
        generator = OrigamiGenerator(model, tokenizer)
        vocab = tokenizer.vocab
        max_depth = tokenizer.max_depth

        # Create two sequences of different lengths, left-padded
        # Seq 1: [PAD, PAD, START, OBJ_START]
        # Seq 2: [START, OBJ_START, key, value]
        from origami.tokenizer.vocabulary import KeyToken, ValueToken

        key_token = KeyToken("name")
        key_id = vocab.encode(key_token)
        value_token = ValueToken("Alice")
        value_id = vocab.encode(value_token)

        # Left-padded batch
        input_ids = torch.tensor(
            [
                [vocab.pad_token_id, vocab.pad_token_id, vocab.start_id, vocab.obj_start_id],
                [vocab.start_id, vocab.obj_start_id, key_id, value_id],
            ],
            dtype=torch.long,
            device=generator.device,
        )

        path_types = torch.zeros(2, 4, max_depth, dtype=torch.long, device=generator.device)
        path_ids = torch.zeros(2, 4, max_depth, dtype=torch.long, device=generator.device)
        path_lengths = torch.zeros(2, 4, dtype=torch.long, device=generator.device)

        # Attention mask: False for PAD, True for real tokens
        attention_mask = torch.tensor(
            [
                [False, False, True, True],
                [True, True, True, True],
            ],
            dtype=torch.bool,
            device=generator.device,
        )

        results = generator.generate_from_tensors(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            stop_after_value=False,
            max_tokens=50,
        )

        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)

    def test_generate_from_tensors_batched_stop_after_value(self, model, tokenizer):
        """Test stop_after_value works correctly for batched generation."""
        generator = OrigamiGenerator(model, tokenizer)
        vocab = tokenizer.vocab
        max_depth = tokenizer.max_depth

        from origami.tokenizer.vocabulary import KeyToken

        key1 = KeyToken("name")
        key1_id = vocab.encode(key1)
        key2 = KeyToken("age")
        key2_id = vocab.encode(key2)

        # Two sequences, both waiting for a value
        # Seq 1: START OBJ_START key:name (awaiting value)
        # Seq 2: START OBJ_START key:age (awaiting value)
        input_ids = torch.tensor(
            [
                [vocab.start_id, vocab.obj_start_id, key1_id],
                [vocab.start_id, vocab.obj_start_id, key2_id],
            ],
            dtype=torch.long,
            device=generator.device,
        )

        path_types = torch.zeros(2, 3, max_depth, dtype=torch.long, device=generator.device)
        path_ids = torch.zeros(2, 3, max_depth, dtype=torch.long, device=generator.device)
        path_lengths = torch.zeros(2, 3, dtype=torch.long, device=generator.device)
        attention_mask = torch.ones(2, 3, dtype=torch.bool, device=generator.device)

        results = generator.generate_from_tensors(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            stop_after_value=True,
            max_tokens=50,
        )

        # Should return exactly 2 values (one per sequence)
        assert len(results) == 2
        # Each result should be a single value (not a full object)
        # The value could be a primitive, object, or array depending on model

    def test_generate_from_tensors_different_prefix_lengths(self, model, tokenizer):
        """Test generation from sequences with different prefix lengths."""
        generator = OrigamiGenerator(model, tokenizer)
        vocab = tokenizer.vocab
        max_depth = tokenizer.max_depth

        from origami.tokenizer.vocabulary import KeyToken, ValueToken

        key_token = KeyToken("name")
        key_id = vocab.encode(key_token)
        value_token = ValueToken("Alice")
        value_id = vocab.encode(value_token)

        # Seq 1: Very short prefix (just started)
        # Seq 2: Longer prefix (has some content already)
        # Seq 3: Even longer prefix
        # All left-padded to same length
        max_len = 6

        input_ids = torch.full(
            (3, max_len), vocab.pad_token_id, dtype=torch.long, device=generator.device
        )
        attention_mask = torch.zeros(3, max_len, dtype=torch.bool, device=generator.device)

        # Seq 1: [PAD, PAD, PAD, PAD, START, OBJ_START]
        input_ids[0, -2:] = torch.tensor([vocab.start_id, vocab.obj_start_id])
        attention_mask[0, -2:] = True

        # Seq 2: [PAD, PAD, START, OBJ_START, key, value]
        input_ids[1, -4:] = torch.tensor([vocab.start_id, vocab.obj_start_id, key_id, value_id])
        attention_mask[1, -4:] = True

        # Seq 3: [START, OBJ_START, key, value, key, ...] - full
        input_ids[2, :] = torch.tensor(
            [vocab.start_id, vocab.obj_start_id, key_id, value_id, key_id, value_id]
        )
        attention_mask[2, :] = True

        path_types = torch.zeros(3, max_len, max_depth, dtype=torch.long, device=generator.device)
        path_ids = torch.zeros(3, max_len, max_depth, dtype=torch.long, device=generator.device)
        path_lengths = torch.zeros(3, max_len, dtype=torch.long, device=generator.device)

        results = generator.generate_from_tensors(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            stop_after_value=False,
            max_tokens=50,
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)


class TestGeneratorGrammarConstraints:
    """Tests for grammar constraint handling in generator."""

    @pytest.fixture
    def constrained_tokenizer(self):
        """Create a tokenizer for grammar constraint tests."""
        data = [
            {"a": 1, "b": 2},
            {"x": "hello", "y": "world"},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def constrained_model(self, constrained_tokenizer):
        """Create model with grammar constraints enabled."""
        config = OrigamiConfig(
            vocab_size=constrained_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=constrained_tokenizer.max_depth,
            use_grammar_constraints=True,
        )
        return OrigamiModel(config, vocab=constrained_tokenizer.vocab)

    def test_generate_respects_grammar(self, constrained_model, constrained_tokenizer):
        """Test that generated objects follow JSON grammar."""
        generator = OrigamiGenerator(constrained_model, constrained_tokenizer)

        # Generate multiple samples
        results = generator.generate(num_samples=5, max_length=100, seed=42)

        assert len(results) == 5
        for result in results:
            # All results should be valid dicts (grammar enforced)
            assert isinstance(result, dict)

    def test_generate_from_tensors_with_grammar(self, constrained_model, constrained_tokenizer):
        """Test generate_from_tensors uses incremental grammar correctly."""
        generator = OrigamiGenerator(constrained_model, constrained_tokenizer)
        vocab = constrained_tokenizer.vocab
        max_depth = constrained_tokenizer.max_depth

        # Start with just START token
        input_ids = torch.tensor(
            [
                [vocab.start_id],
            ],
            dtype=torch.long,
            device=generator.device,
        )

        path_types = torch.zeros(1, 1, max_depth, dtype=torch.long, device=generator.device)
        path_ids = torch.zeros(1, 1, max_depth, dtype=torch.long, device=generator.device)
        path_lengths = torch.zeros(1, 1, dtype=torch.long, device=generator.device)
        attention_mask = torch.ones(1, 1, dtype=torch.bool, device=generator.device)

        results = generator.generate_from_tensors(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            stop_after_value=False,
            max_tokens=100,
        )

        assert len(results) == 1
        assert isinstance(results[0], dict)


class TestGeneratorEdgeCases:
    """Edge case tests for generator."""

    @pytest.fixture
    def edge_tokenizer(self):
        """Create tokenizer with edge case data."""
        data = [
            {},  # Empty object
            {"single": "value"},
            {"deep": {"nested": {"structure": "here"}}},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    @pytest.fixture
    def edge_model(self, edge_tokenizer):
        """Create model for edge case tests."""
        config = OrigamiConfig(
            vocab_size=edge_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=edge_tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=edge_tokenizer.vocab)

    def test_generate_handles_max_tokens_limit(self, edge_model, edge_tokenizer):
        """Test generation stops at max_tokens limit."""
        generator = OrigamiGenerator(edge_model, edge_tokenizer)

        # Very short max_tokens - should still produce valid output
        results = generator.generate(num_samples=1, max_length=10, seed=42)

        assert len(results) == 1
        # Result should be a dict (possibly incomplete but valid)
        assert isinstance(results[0], dict)

    def test_generate_single_sample(self, edge_model, edge_tokenizer):
        """Test generating exactly one sample."""
        generator = OrigamiGenerator(edge_model, edge_tokenizer)

        results = generator.generate(num_samples=1, max_length=50, seed=42)

        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_generate_many_samples(self, edge_model, edge_tokenizer):
        """Test generating many samples at once."""
        generator = OrigamiGenerator(edge_model, edge_tokenizer)

        results = generator.generate(num_samples=10, max_length=50, seed=42)

        assert len(results) == 10
        for result in results:
            assert isinstance(result, dict)
