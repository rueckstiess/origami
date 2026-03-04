"""Integration tests for continuous head functionality."""

import math

import pytest
import torch

from origami.config import ModelConfig
from origami.model import OrigamiModel
from origami.model.heads import ContinuousHead
from origami.preprocessing import NumericScaler, ScaledNumeric
from origami.tokenizer import JSONTokenizer
from origami.training import OrigamiDataCollator


class TestContinuousHeadConfig:
    """Tests for continuous head configuration."""

    def test_config_defaults(self):
        """Test default continuous head config."""
        config = ModelConfig()
        assert config.use_continuous_head is False
        assert config.num_mixture_components == 5
        assert config.continuous_loss_weight == -1.0  # Auto

    def test_config_enabled(self):
        """Test enabling continuous head."""
        config = ModelConfig(
            use_continuous_head=True,
            num_mixture_components=3,
            continuous_loss_weight=0.5,
        )
        assert config.use_continuous_head is True
        assert config.num_mixture_components == 3
        assert config.continuous_loss_weight == 0.5


class TestEmbeddingsWithNumeric:
    """Tests for embeddings with numeric values."""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer fitted on data with ScaledNumeric."""
        data = [
            {"name": "item", "price": ScaledNumeric(0.5)},
            {"name": "thing", "price": ScaledNumeric(-0.3)},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    def test_embeddings_have_num_embedding(self, tokenizer):
        """Test embeddings module has num_embedding when enabled."""
        config = ModelConfig(
            use_continuous_head=True,
            d_model=64,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        assert hasattr(model.embeddings, "num_embedding")
        assert model.embeddings.num_embedding.shape == (64,)

    def test_embeddings_no_num_embedding_when_disabled(self, tokenizer):
        """Test embeddings don't have num_embedding when disabled."""
        config = ModelConfig(
            use_continuous_head=False,
            d_model=64,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        assert not hasattr(model.embeddings, "num_embedding")

    def test_num_embedding_affects_output(self, tokenizer):
        """Test that numeric values affect embedding output."""
        config = ModelConfig(
            use_continuous_head=True,
            d_model=64,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        model.eval()

        # Create batch with NUM token
        data = [{"name": "test", "price": ScaledNumeric(0.5)}]
        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        # Forward with different numeric values
        with torch.no_grad():
            output1 = model.embeddings(
                batch.input_ids,
                batch.path_types,
                batch.path_ids,
                batch.path_lengths,
                numeric_values=batch.numeric_values,
            )

            # Change numeric value
            modified_numerics = batch.numeric_values.clone()
            modified_numerics[batch.numeric_mask] = 2.0

            output2 = model.embeddings(
                batch.input_ids,
                batch.path_types,
                batch.path_ids,
                batch.path_lengths,
                numeric_values=modified_numerics,
            )

        # Outputs should differ at NUM token positions
        diff = (output1 - output2).abs()
        assert diff[batch.numeric_mask].sum() > 0


class TestModelWithContinuousHead:
    """Tests for model with continuous head enabled."""

    @pytest.fixture
    def setup(self):
        """Create model and tokenizer for continuous head testing."""
        # Create data with scaled numerics
        data = [
            {"name": "a", "value": ScaledNumeric(0.1)},
            {"name": "b", "value": ScaledNumeric(0.5)},
            {"name": "c", "value": ScaledNumeric(-0.2)},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(
            use_continuous_head=True,
            num_mixture_components=3,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        return model, tokenizer, data

    def test_forward_returns_continuous_params(self, setup):
        """Test forward returns continuous parameters."""
        model, tokenizer, data = setup

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        output = model(
            batch.input_ids,
            batch.path_types,
            batch.path_ids,
            batch.path_lengths,
            batch.attention_mask,
        )

        assert output.continuous_params is not None
        weights, means, log_vars = output.continuous_params

        # Check shapes
        batch_size, seq_len = batch.input_ids.shape
        num_components = model.config.num_mixture_components

        assert weights.shape == (batch_size, seq_len, num_components)
        assert means.shape == (batch_size, seq_len, num_components)
        assert log_vars.shape == (batch_size, seq_len, num_components)

    def test_continuous_loss_computed(self, setup):
        """Test continuous loss is computed when labels provided."""
        model, tokenizer, data = setup

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        # Forward with labels - model handles shift internally
        output = model(
            batch.input_ids,
            batch.path_types,
            batch.path_ids,
            batch.path_lengths,
            batch.attention_mask,
            labels=batch.labels,
            numeric_values=batch.numeric_values,
            numeric_mask=batch.numeric_mask,
        )

        assert output.loss is not None
        assert output.loss.item() > 0

    def test_gradient_flows_through_continuous_head(self, setup):
        """Test gradients flow through continuous head."""
        model, tokenizer, data = setup
        model.train()

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        output = model(
            batch.input_ids,
            batch.path_types,
            batch.path_ids,
            batch.path_lengths,
            batch.attention_mask,
            labels=batch.labels,
            numeric_values=batch.numeric_values,
            numeric_mask=batch.numeric_mask,
        )

        output.loss.backward()

        # Check continuous head has gradients
        for param in model.continuous_head.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0


class TestCollatorWithNumerics:
    """Tests for collator with numeric values."""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer fitted on data with ScaledNumeric."""
        data = [
            {"x": ScaledNumeric(0.1), "y": "a"},
            {"x": ScaledNumeric(0.5), "y": "b"},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        return tokenizer

    def test_collator_produces_numeric_tensors(self, tokenizer):
        """Test collator produces numeric_values and numeric_mask."""
        data = [
            {"x": ScaledNumeric(0.3), "y": "test"},
            {"x": ScaledNumeric(-0.1), "y": "test"},
        ]
        instances = [tokenizer.tokenize(obj) for obj in data]

        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        assert batch.numeric_values is not None
        assert batch.numeric_mask is not None

    def test_numeric_mask_marks_num_positions(self, tokenizer):
        """Test numeric_mask marks NUM token positions."""
        data = [{"x": ScaledNumeric(0.5), "y": "test"}]
        instances = [tokenizer.tokenize(obj) for obj in data]

        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        # Find NUM token positions
        num_token_id = 9  # NUM token ID
        num_positions = batch.input_ids == num_token_id

        # numeric_mask should match NUM positions
        assert (batch.numeric_mask == num_positions).all()

    def test_numeric_values_at_correct_positions(self, tokenizer):
        """Test numeric values are at correct positions."""
        data = [{"x": ScaledNumeric(0.75), "y": "a"}]
        instances = [tokenizer.tokenize(obj) for obj in data]

        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        # Get value at NUM position
        num_pos = batch.numeric_mask[0].nonzero(as_tuple=True)[0]
        if len(num_pos) > 0:
            value = batch.numeric_values[0, num_pos[0]].item()
            assert abs(value - 0.75) < 0.001


class TestEndToEndContinuousPipeline:
    """End-to-end tests for continuous head pipeline."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline with continuous head."""
        # 1. Create high-cardinality data
        data = [{"price": float(i * 10), "category": i % 3} for i in range(200)]

        # 2. Scale with NumericScaler
        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        scaled_data = scaler.transform(data)

        # 3. Tokenize
        tokenizer = JSONTokenizer()
        tokenizer.fit(scaled_data)

        # 4. Create model with continuous head
        config = ModelConfig(
            use_continuous_head=True,
            num_mixture_components=3,
            d_model=32,
            n_heads=2,
            n_layers=2,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        model.train()

        # 5. Create batch and train step
        instances = [tokenizer.tokenize(obj) for obj in scaled_data[:8]]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training step - model handles shift internally
        output = model(
            batch.input_ids,
            batch.path_types,
            batch.path_ids,
            batch.path_lengths,
            batch.attention_mask,
            labels=batch.labels,
            numeric_values=batch.numeric_values,
            numeric_mask=batch.numeric_mask,
        )

        _initial_loss = output.loss.item()  # noqa: F841
        output.loss.backward()
        optimizer.step()

        # Another step should reduce loss (usually)
        optimizer.zero_grad()
        output2 = model(
            batch.input_ids,
            batch.path_types,
            batch.path_ids,
            batch.path_lengths,
            batch.attention_mask,
            labels=batch.labels,
            numeric_values=batch.numeric_values,
            numeric_mask=batch.numeric_mask,
        )

        # Just verify training runs without error
        assert output2.loss is not None

    def test_loss_weight_auto_calculation(self):
        """Test auto loss weight calculation based on NUM token proportion."""
        # Create data where ~10% of tokens are NUM
        data = [{"val": ScaledNumeric(0.5), "a": "x", "b": "y", "c": "z"}]

        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(
            use_continuous_head=True,
            continuous_loss_weight=-1.0,  # Auto
            d_model=32,
            n_heads=2,
            n_layers=2,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        # Verify auto weight is applied (loss should compute without error)
        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        output = model(
            batch.input_ids,
            batch.path_types,
            batch.path_ids,
            batch.path_lengths,
            batch.attention_mask,
            labels=batch.labels,
            numeric_values=batch.numeric_values,
            numeric_mask=batch.numeric_mask,
        )

        assert output.loss is not None


class TestMixedDiscreteAndContinuous:
    """Tests for data with both discrete and continuous fields."""

    def test_mixed_data_processing(self):
        """Test processing data with both discrete and continuous fields."""
        # Data with low-cardinality (discrete) and high-cardinality (continuous)
        data = [{"category": "A", "amount": float(i * 100)} for i in range(150)]

        # Scale
        scaler = NumericScaler(cat_threshold=100)
        scaler.fit(data)
        scaled = scaler.transform(data)

        # Verify category is unchanged, amount is scaled
        assert scaled[0]["category"] == "A"
        assert isinstance(scaled[0]["amount"], ScaledNumeric)

        # Tokenize
        tokenizer = JSONTokenizer()
        tokenizer.fit(scaled)

        instance = tokenizer.tokenize(scaled[0])

        # Should have both regular tokens and NUM tokens
        from origami.tokenizer.vocabulary import NUM

        tokens_have_num = any(str(t) == str(NUM) for t in instance.tokens)
        assert tokens_have_num

    def test_no_scaled_numerics_disables_continuous(self):
        """Test that data without ScaledNumeric works with disabled continuous head."""
        data = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]

        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(
            use_continuous_head=False,
            d_model=32,
            n_heads=2,
            n_layers=2,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        # Should work without numeric tensors
        output = model(
            batch.input_ids,
            batch.path_types,
            batch.path_ids,
            batch.path_lengths,
            batch.attention_mask,
            labels=batch.labels,
        )

        assert output.loss is not None
        assert output.continuous_params is None


class TestTruncatedMoG:
    """Tests for truncated MoG NLL and discretized NLL."""

    @pytest.fixture
    def head(self):
        config = ModelConfig(use_continuous_head=True, num_mixture_components=5, d_model=32)
        return ContinuousHead(config)

    def test_nll_no_bounds_matches_standard(self, head):
        """Without bounds, truncated path should match standard path."""
        torch.manual_seed(42)
        batch, seq, K = 2, 4, head.n_components
        weights = torch.softmax(torch.randn(batch, seq, K), dim=-1)
        means = torch.randn(batch, seq, K)
        log_vars = torch.randn(batch, seq, K) * 0.5
        targets = torch.randn(batch, seq)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        # Standard path (no bounds at all)
        loss_std = head.nll_loss(weights, means, log_vars, targets, mask)
        # Truncated path with ±inf bounds (should be equivalent)
        lower = torch.full((batch, seq), float("-inf"))
        upper = torch.full((batch, seq), float("inf"))
        loss_trunc = head.nll_loss(
            weights, means, log_vars, targets, mask,
            lower=lower, upper=upper,
        )
        torch.testing.assert_close(loss_std, loss_trunc, atol=1e-4, rtol=1e-4)

    def test_nll_with_bounds_lower_than_unbounded(self, head):
        """Truncated NLL should be <= unbounded NLL on data within bounds."""
        torch.manual_seed(42)
        batch, seq, K = 4, 6, head.n_components
        weights = torch.softmax(torch.randn(batch, seq, K), dim=-1)
        means = torch.randn(batch, seq, K) + 2.0  # bias means positive
        log_vars = torch.zeros(batch, seq, K)
        # Non-negative targets
        targets = torch.rand(batch, seq) * 5
        mask = torch.ones(batch, seq, dtype=torch.bool)

        loss_unbounded = head.nll_loss(weights, means, log_vars, targets, mask)
        lower = torch.zeros(batch, seq)
        upper = torch.full((batch, seq), 10.0)
        loss_bounded = head.nll_loss(
            weights, means, log_vars, targets, mask,
            lower=lower, upper=upper,
        )
        # Bounded NLL should be lower (more probability mass on valid range)
        assert loss_bounded.item() <= loss_unbounded.item() + 1e-5

    def test_nll_integer_discretized(self, head):
        """Discretized NLL should give proper probabilities for integers."""
        torch.manual_seed(42)
        batch, seq, K = 1, 3, head.n_components
        weights = torch.softmax(torch.randn(batch, seq, K), dim=-1)
        means = torch.randn(batch, seq, K)
        log_vars = torch.zeros(batch, seq, K)
        targets = torch.tensor([[0.0, 1.0, 2.0]])
        mask = torch.ones(batch, seq, dtype=torch.bool)
        lower = torch.zeros(batch, seq)
        upper = torch.full((batch, seq), 10.0)
        is_integer = torch.ones(batch, seq, dtype=torch.bool)

        loss = head.nll_loss(
            weights, means, log_vars, targets, mask,
            lower=lower, upper=upper, is_integer=is_integer,
        )
        assert loss.isfinite()
        assert loss.item() > 0

    def test_nll_mixed_integer_and_continuous(self, head):
        """Mix of integer and continuous positions in same batch."""
        torch.manual_seed(42)
        batch, seq, K = 2, 4, head.n_components
        weights = torch.softmax(torch.randn(batch, seq, K), dim=-1)
        means = torch.randn(batch, seq, K)
        log_vars = torch.zeros(batch, seq, K)
        targets = torch.randn(batch, seq).abs()
        mask = torch.ones(batch, seq, dtype=torch.bool)
        lower = torch.zeros(batch, seq)
        upper = torch.full((batch, seq), 10.0)
        # Only first two positions per batch are integer
        is_integer = torch.zeros(batch, seq, dtype=torch.bool)
        is_integer[:, :2] = True

        loss = head.nll_loss(
            weights, means, log_vars, targets, mask,
            lower=lower, upper=upper, is_integer=is_integer,
        )
        assert loss.isfinite()

    def test_gradient_flow_through_truncated_nll(self, head):
        """Gradients flow through truncation normalization."""
        batch, seq, K = 2, 3, head.n_components
        hidden = torch.randn(batch, seq, head.d_model, requires_grad=True)
        weights, means, log_vars = head(hidden)
        targets = torch.rand(batch, seq) * 5
        mask = torch.ones(batch, seq, dtype=torch.bool)
        lower = torch.zeros(batch, seq)
        upper = torch.full((batch, seq), 10.0)
        is_integer = torch.ones(batch, seq, dtype=torch.bool)

        loss = head.nll_loss(
            weights, means, log_vars, targets, mask,
            lower=lower, upper=upper, is_integer=is_integer,
        )
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.abs().sum() > 0

    def test_gradient_no_nan_with_inf_bounds(self, head):
        """Gradients must be finite when bounds contain ±inf (non-numeric positions).

        Regression test: ±inf bounds caused 0 * inf = NaN in autograd because
        the CDF gradient PDF(inf) * (-inf/std²) is 0 * inf = NaN. Fixed by
        clamping bounds to finite values in _log_mixture_prob.
        """
        batch, seq, K = 2, 6, head.n_components
        hidden = torch.randn(batch, seq, head.d_model, requires_grad=True)
        weights, means, log_vars = head(hidden)
        targets = torch.rand(batch, seq)
        # Only some positions are numeric
        mask = torch.zeros(batch, seq, dtype=torch.bool)
        mask[:, 2] = True  # Only position 2 is numeric

        # Bounds: finite at numeric positions, ±inf elsewhere (like collator produces)
        lower = torch.full((batch, seq), float("-inf"))
        upper = torch.full((batch, seq), float("inf"))
        lower[:, 2] = 0.0
        upper[:, 2] = 1.0
        is_integer = torch.zeros(batch, seq, dtype=torch.bool)
        is_integer[:, 2] = True

        loss = head.nll_loss(
            weights, means, log_vars, targets, mask,
            lower=lower, upper=upper, is_integer=is_integer,
        )
        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        loss.backward()
        assert hidden.grad is not None
        assert not torch.isnan(hidden.grad).any(), "NaN in gradients with ±inf bounds"
        assert torch.isfinite(hidden.grad).all(), "Inf in gradients with ±inf bounds"

    def test_gradient_no_nan_with_all_inf_bounds(self, head):
        """Gradients must be finite even when ALL bounds are ±inf."""
        batch, seq, K = 2, 4, head.n_components
        hidden = torch.randn(batch, seq, head.d_model, requires_grad=True)
        weights, means, log_vars = head(hidden)
        targets = torch.randn(batch, seq)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        lower = torch.full((batch, seq), float("-inf"))
        upper = torch.full((batch, seq), float("inf"))

        loss = head.nll_loss(
            weights, means, log_vars, targets, mask,
            lower=lower, upper=upper,
        )
        assert loss.isfinite()
        loss.backward()
        assert not torch.isnan(hidden.grad).any(), "NaN in gradients with all-inf bounds"

    def test_collator_bounds_default_to_none_without_schema(self):
        """Without schema_pda, bounds tensors should be None."""
        data = [{"x": ScaledNumeric(0.5), "y": "a"}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        collator = OrigamiDataCollator(tokenizer)  # No schema_pda
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        assert batch.numeric_lower is None
        assert batch.numeric_upper is None
        assert batch.is_integer is None

    def test_collator_produces_bounds_with_schema(self):
        """With schema_pda, bounds are populated at NUM positions."""
        from origami.constraints.schema_pda import SchemaPDA

        data = [{"count": ScaledNumeric(0.5), "name": "test"}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 0, "maximum": 100},
                "name": {"type": "string"},
            },
        }
        schema_pda = SchemaPDA(schema, tokenizer.vocab, max_depth=32)
        collator = OrigamiDataCollator(tokenizer, schema_pda=schema_pda)

        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        assert batch.numeric_lower is not None
        assert batch.numeric_upper is not None
        assert batch.is_integer is not None

        # At NUM positions, bounds should be 0 and 100
        num_positions = batch.numeric_mask
        if num_positions.any():
            assert (batch.numeric_lower[num_positions] == 0.0).all()
            assert (batch.numeric_upper[num_positions] == 100.0).all()
            assert batch.is_integer[num_positions].all()

        # At non-continuous positions, bounds should be -inf/+inf
        non_num = ~num_positions & batch.attention_mask
        if non_num.any():
            assert (batch.numeric_lower[non_num] == float("-inf")).all()
            assert (batch.numeric_upper[non_num] == float("inf")).all()
            assert not batch.is_integer[non_num].any()


class TestArrayLengthModeling:
    """Tests for array length modeling via the continuous head."""

    def test_tokenizer_stores_array_length(self):
        """Tokenizer stores array length in numeric_values at ARRAY_START."""
        data = [{"items": [1, 2, 3], "name": "test"}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        inst = tokenizer.tokenize(data[0])

        from origami.tokenizer.vocabulary import ARRAY_START
        array_start_indices = [
            i for i, t in enumerate(inst.tokens) if t == ARRAY_START
        ]
        assert len(array_start_indices) == 1
        idx = array_start_indices[0]
        assert inst.numeric_values[idx] == 3.0

    def test_tokenizer_empty_array_length(self):
        """Tokenizer stores 0.0 for empty arrays."""
        data = [{"items": [], "name": "test"}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        inst = tokenizer.tokenize(data[0])

        from origami.tokenizer.vocabulary import ARRAY_START
        array_start_indices = [
            i for i, t in enumerate(inst.tokens) if t == ARRAY_START
        ]
        assert len(array_start_indices) == 1
        assert inst.numeric_values[array_start_indices[0]] == 0.0

    def test_tokenizer_nested_arrays(self):
        """Tokenizer stores correct lengths for nested arrays."""
        data = [{"matrix": [[1, 2], [3, 4, 5]]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        inst = tokenizer.tokenize(data[0])

        from origami.tokenizer.vocabulary import ARRAY_START
        array_lengths = [
            inst.numeric_values[i]
            for i, t in enumerate(inst.tokens)
            if t == ARRAY_START
        ]
        assert sorted(array_lengths) == [2.0, 2.0, 3.0]

    def test_collator_normalizes_array_length(self):
        """Collator normalizes array lengths to [0, 1] range."""
        from origami.constraints.schema_pda import SchemaPDA

        data = [{"items": [1, 2, 3]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": 10,
                },
            },
        }
        schema_pda = SchemaPDA(schema, tokenizer.vocab, max_depth=32)
        collator = OrigamiDataCollator(tokenizer, schema_pda=schema_pda)
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        # Find ARRAY_START position
        array_start_pos = (batch.input_ids == 4).nonzero(as_tuple=False)
        assert len(array_start_pos) == 1
        b, pos = array_start_pos[0].tolist()

        # Should be normalized: 3 / 10 = 0.3
        assert abs(batch.numeric_values[b, pos].item() - 0.3) < 1e-5
        assert batch.numeric_mask[b, pos]

    def test_collator_array_bounds_normalized(self):
        """Collator sets normalized bounds for array lengths."""
        from origami.constraints.schema_pda import SchemaPDA

        data = [{"items": [1, 2]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": 10,
                },
            },
        }
        schema_pda = SchemaPDA(schema, tokenizer.vocab, max_depth=32)
        collator = OrigamiDataCollator(tokenizer, schema_pda=schema_pda)
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        array_start_pos = (batch.input_ids == 4).nonzero(as_tuple=False)
        b, pos = array_start_pos[0].tolist()

        # Bounds normalized: lower = 1/10 = 0.1, upper = 10/10 = 1.0
        assert abs(batch.numeric_lower[b, pos].item() - 0.1) < 1e-5
        assert abs(batch.numeric_upper[b, pos].item() - 1.0) < 1e-5
        assert batch.is_integer[b, pos]

    def test_embeddings_handle_array_start(self):
        """Embedding output changes when array length changes."""
        data = [{"items": [1, 2, 3]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(use_continuous_head=True, d_model=64)
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        model.eval()

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        with torch.no_grad():
            out1 = model.embeddings(
                batch.input_ids, batch.path_types, batch.path_ids,
                batch.path_lengths, numeric_values=batch.numeric_values,
            )
            modified = batch.numeric_values.clone()
            # Change array length value at ARRAY_START position
            is_arr = batch.input_ids == 4
            modified[is_arr] = 10.0
            out2 = model.embeddings(
                batch.input_ids, batch.path_types, batch.path_ids,
                batch.path_lengths, numeric_values=modified,
            )

        diff = (out1 - out2).abs()
        assert diff[is_arr].sum() > 0

    def test_loss_includes_array_length_positions(self):
        """Model loss is computed at ARRAY_START positions."""
        data = [
            {"items": [1, 2], "val": ScaledNumeric(0.5)},
            {"items": [3], "val": ScaledNumeric(-0.1)},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(
            use_continuous_head=True, num_mixture_components=3,
            d_model=32, n_heads=2, n_layers=2,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        model.train()

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer)
        batch = collator(instances)

        # Verify ARRAY_START is in numeric_mask
        is_arr = batch.input_ids == 4
        assert (batch.numeric_mask & is_arr).any()

        output = model(
            batch.input_ids, batch.path_types, batch.path_ids,
            batch.path_lengths, batch.attention_mask,
            labels=batch.labels,
            numeric_values=batch.numeric_values,
            numeric_mask=batch.numeric_mask,
        )
        assert output.loss is not None
        assert output.loss.isfinite()
