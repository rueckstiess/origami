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
    """Tests for MoG NLL and truncated sampling."""

    @pytest.fixture
    def head(self):
        config = ModelConfig(use_continuous_head=True, num_mixture_components=5, d_model=32)
        return ContinuousHead(config)

    def test_truncated_sampling_respects_bounds(self, head):
        """Truncated sampling produces values within [lower, upper]."""
        torch.manual_seed(42)
        batch, seq, K = 8, 1, head.n_components
        weights = torch.softmax(torch.randn(batch, seq, K), dim=-1)
        means = torch.randn(batch, seq, K)
        log_vars = torch.zeros(batch, seq, K)
        lower = torch.zeros(batch, seq)
        upper = torch.ones(batch, seq)

        samples = head.sample(weights, means, log_vars, lower=lower, upper=upper)
        assert (samples >= 0).all(), f"Samples below lower bound: {samples.min()}"
        assert (samples <= 1).all(), f"Samples above upper bound: {samples.max()}"


class TestDiscretizedLogisticNLL:
    """Tests for the discretized logistic mixture NLL used for integer positions."""

    @pytest.fixture
    def head(self):
        config = ModelConfig(use_continuous_head=True, num_mixture_components=5, d_model=32)
        return ContinuousHead(config)

    def test_interior_bin(self, head):
        """Single component at target: verify NLL matches manual sigmoid computation."""

        # 1 component, target=0.5, mean=0.5, scale=0.1, step=0.2
        weights = torch.ones(1, 1, 1)
        means = torch.tensor([[[0.5]]])
        log_vars = torch.tensor([[[2 * math.log(0.1)]]])  # scale = exp(0.5*lv) = 0.1
        targets = torch.tensor([[0.5]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        is_integer = torch.ones(1, 1, dtype=torch.bool)
        step = torch.tensor([[0.2]])

        loss = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            is_integer=is_integer,
            discretization_step=step,
        )

        # Manual: sigmoid((0.6 - 0.5)/0.1) - sigmoid((0.4 - 0.5)/0.1)
        #       = sigmoid(1.0) - sigmoid(-1.0)
        p = torch.sigmoid(torch.tensor(1.0)) - torch.sigmoid(torch.tensor(-1.0))
        expected_nll = -math.log(p.item())
        assert abs(loss.item() - expected_nll) < 0.01

    def test_lower_boundary_absorbs_mass(self, head):
        """Target at lower boundary (0.0): all mass below is absorbed."""

        weights = torch.ones(1, 1, 1)
        means = torch.tensor([[[0.0]]])
        log_vars = torch.zeros(1, 1, 1)  # scale = 1.0
        targets = torch.tensor([[0.0]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        is_integer = torch.ones(1, 1, dtype=torch.bool)
        step = torch.tensor([[0.2]])  # half_step = 0.1

        loss = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            is_integer=is_integer,
            discretization_step=step,
        )

        # Lower boundary: sigmoid((0.1 - 0)/1.0) - sigmoid(-20) ≈ sigmoid(0.1)
        expected = -math.log(torch.sigmoid(torch.tensor(0.1)).item())
        assert abs(loss.item() - expected) < 0.01
        assert loss.isfinite()

    def test_upper_boundary_absorbs_mass(self, head):
        """Target at upper boundary (1.0): all mass above is absorbed."""

        weights = torch.ones(1, 1, 1)
        means = torch.tensor([[[1.0]]])
        log_vars = torch.zeros(1, 1, 1)  # scale = 1.0
        targets = torch.tensor([[1.0]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        is_integer = torch.ones(1, 1, dtype=torch.bool)
        step = torch.tensor([[0.2]])

        loss = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            is_integer=is_integer,
            discretization_step=step,
        )
        # Upper boundary: sigmoid(20) - sigmoid((0.9 - 1.0)/1.0) = 1 - sigmoid(-0.1) = sigmoid(0.1)
        expected = -math.log(torch.sigmoid(torch.tensor(0.1)).item())
        assert abs(loss.item() - expected) < 0.01
        assert loss.isfinite()

    def test_no_nan_extreme_means(self, head):
        """No NaN even when component means are far from target."""
        weights = torch.ones(1, 1, 1)
        means = torch.tensor([[[100.0]]])
        log_vars = torch.zeros(1, 1, 1)
        targets = torch.tensor([[0.5]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        is_integer = torch.ones(1, 1, dtype=torch.bool)
        step = torch.tensor([[0.1]])

        loss = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            is_integer=is_integer,
            discretization_step=step,
        )
        assert loss.isfinite()

    def test_numerical_stability_tiny_scale(self, head):
        """Very small scale (sharp distribution) doesn't produce NaN."""
        weights = torch.ones(1, 1, 1)
        means = torch.tensor([[[0.5]]])
        log_vars = torch.tensor([[[-20.0]]])  # scale ≈ 4.5e-5
        targets = torch.tensor([[0.5]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        is_integer = torch.ones(1, 1, dtype=torch.bool)
        step = torch.tensor([[0.1]])

        loss = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            is_integer=is_integer,
            discretization_step=step,
        )
        assert loss.isfinite()

    def test_numerical_stability_large_scale(self, head):
        """Very large scale (flat distribution) doesn't produce NaN."""
        weights = torch.ones(1, 1, 1)
        means = torch.tensor([[[0.5]]])
        log_vars = torch.tensor([[[20.0]]])  # scale ≈ 22026
        targets = torch.tensor([[0.5]])
        mask = torch.ones(1, 1, dtype=torch.bool)
        is_integer = torch.ones(1, 1, dtype=torch.bool)
        step = torch.tensor([[0.1]])

        loss = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            is_integer=is_integer,
            discretization_step=step,
        )
        assert loss.isfinite()

    def test_backward_compat_no_is_integer(self, head):
        """When is_integer is None, behavior identical to standard Gaussian NLL."""
        torch.manual_seed(42)
        weights = torch.softmax(torch.randn(2, 4, 5), dim=-1)
        means = torch.randn(2, 4, 5)
        log_vars = torch.randn(2, 4, 5)
        targets = torch.randn(2, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)

        loss_old = head._nll_loss_standard(weights, means, log_vars, targets, mask, None)
        loss_new = head.nll_loss(weights, means, log_vars, targets, mask)
        assert torch.allclose(loss_old, loss_new)

    def test_mixed_integer_float_positions(self, head):
        """Mixed batch: integer positions use logistic, float positions use Gaussian."""
        torch.manual_seed(42)
        batch, seq, K = 1, 4, head.n_components
        weights = torch.softmax(torch.randn(batch, seq, K), dim=-1)
        means = torch.randn(batch, seq, K) * 0.5
        log_vars = torch.randn(batch, seq, K) * 0.5
        targets = torch.rand(batch, seq)
        mask = torch.ones(batch, seq, dtype=torch.bool)
        is_integer = torch.tensor([[True, False, True, False]])
        step = torch.full((batch, seq), 0.1)

        loss = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            is_integer=is_integer,
            discretization_step=step,
        )
        assert loss.isfinite()

        # Verify it's a weighted combination
        gauss_loss = head._nll_loss_standard(
            weights,
            means,
            log_vars,
            targets,
            mask & ~is_integer,
            None,
        )
        logistic_loss = head._nll_loss_discretized_logistic(
            weights,
            means,
            log_vars,
            targets,
            mask & is_integer,
            step,
            None,
        )
        expected = (2 * gauss_loss + 2 * logistic_loss) / 4
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients flow correctly through the discretized logistic loss."""
        config = ModelConfig(
            use_continuous_head=True,
            num_mixture_components=3,
            d_model=32,
            n_heads=2,
            n_layers=2,
        )
        data = [
            {"items": [1, 2], "val": ScaledNumeric(0.5)},
            {"items": [3], "val": ScaledNumeric(-0.1)},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        model.train()

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer, model_array_lengths=True)
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
            is_integer=batch.is_integer,
            discretization_step=batch.discretization_step,
        )
        assert output.loss.isfinite()
        output.loss.backward()

        for param in model.continuous_head.parameters():
            assert param.grad is not None
            assert param.grad.isfinite().all()

    def test_collator_sets_is_integer_for_array_start(self):
        """Collator populates is_integer=True and correct step at ARRAY_START.

        Normalization comes from the data-derived array_max_lengths map, not the
        schema, so the divisor is decoupled from schema masking.
        """
        data = [{"items": [1, 2, 3]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        collator = OrigamiDataCollator(
            tokenizer,
            model_array_lengths=True,
            array_max_lengths={"items": 10},
        )
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        from origami.preprocessing import array_length_norm
        from origami.tokenizer.path import KeyElement

        norm = array_length_norm((KeyElement("items"),), {"items": 10})
        array_start_pos = batch.input_ids == 4
        assert batch.is_integer is not None
        assert batch.is_integer[array_start_pos].all()
        # Step is 1 / (buffered) norm
        assert batch.discretization_step[array_start_pos].allclose(torch.tensor(1 / norm))
        # Non-ARRAY_START positions should be False/0
        assert not batch.is_integer[~array_start_pos].any()

    def test_collator_no_is_integer_without_arrays(self):
        """Without arrays, is_integer has no True values."""
        data = [{"x": ScaledNumeric(0.5), "y": "a"}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        collator = OrigamiDataCollator(tokenizer)
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)
        assert not batch.is_integer.any()

    def test_collator_skips_array_start_when_model_array_lengths_false(self):
        """Collator does NOT set numeric values for ARRAY_START when model_array_lengths=False."""
        data = [{"items": [1, 2, 3]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        # Default: model_array_lengths=False
        collator = OrigamiDataCollator(tokenizer)
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        array_start_pos = batch.input_ids == 4
        # No numeric mask at ARRAY_START
        assert not (batch.numeric_mask & array_start_pos).any()
        # No is_integer at ARRAY_START
        assert not batch.is_integer[array_start_pos].any()
        # No discretization_step at ARRAY_START
        assert batch.discretization_step[array_start_pos].sum() == 0

    def test_collator_preserves_numeric_values_when_model_array_lengths_false(self):
        """Non-ARRAY_START numeric values are preserved when model_array_lengths=False."""
        data = [{"items": [1, 2], "val": ScaledNumeric(0.5)}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        collator = OrigamiDataCollator(tokenizer, model_array_lengths=False)
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        # ScaledNumeric should still have numeric values
        assert batch.numeric_mask.any()
        # But not at ARRAY_START positions
        array_start_pos = batch.input_ids == 4
        assert not (batch.numeric_mask & array_start_pos).any()


class TestArrayLengthModeling:
    """Tests for array length modeling via the continuous head."""

    def test_tokenizer_stores_array_length(self):
        """Tokenizer stores array length in numeric_values at ARRAY_START."""
        data = [{"items": [1, 2, 3], "name": "test"}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        inst = tokenizer.tokenize(data[0])

        from origami.tokenizer.vocabulary import ARRAY_START

        array_start_indices = [i for i, t in enumerate(inst.tokens) if t == ARRAY_START]
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

        array_start_indices = [i for i, t in enumerate(inst.tokens) if t == ARRAY_START]
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
            inst.numeric_values[i] for i, t in enumerate(inst.tokens) if t == ARRAY_START
        ]
        assert sorted(array_lengths) == [2.0, 2.0, 3.0]

    def test_collator_normalizes_array_length(self):
        """Collator normalizes array lengths to [0, 1] range.

        The divisor comes from the data-derived array_max_lengths map, not the
        schema, so normalization is independent of schema masking.
        """
        data = [{"items": [1, 2, 3]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        collator = OrigamiDataCollator(
            tokenizer,
            model_array_lengths=True,
            array_max_lengths={"items": 10},
        )
        instances = [tokenizer.tokenize(obj) for obj in data]
        batch = collator(instances)

        from origami.preprocessing import array_length_norm
        from origami.tokenizer.path import KeyElement

        # Find ARRAY_START position
        array_start_pos = (batch.input_ids == 4).nonzero(as_tuple=False)
        assert len(array_start_pos) == 1
        b, pos = array_start_pos[0].tolist()

        # Normalized by the buffered divisor (3 / norm)
        norm = array_length_norm((KeyElement("items"),), {"items": 10})
        assert abs(batch.numeric_values[b, pos].item() - 3 / norm) < 1e-5
        assert batch.numeric_mask[b, pos]

    def test_embeddings_array_start_uses_learned_token(self):
        """ARRAY_START uses its learned token embedding, not multiplicative numeric embedding.

        Array length is predicted by the continuous head from context but the
        embedding doesn't encode it — length is enforced via mask overrides
        during generation. This avoids the zero-embedding problem for empty arrays.
        """
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
                batch.input_ids,
                batch.path_types,
                batch.path_ids,
                batch.path_lengths,
                numeric_values=batch.numeric_values,
            )
            modified = batch.numeric_values.clone()
            # Change array length value at ARRAY_START position
            is_arr = batch.input_ids == 4
            modified[is_arr] = 10.0
            out2 = model.embeddings(
                batch.input_ids,
                batch.path_types,
                batch.path_ids,
                batch.path_lengths,
                numeric_values=modified,
            )

        diff = (out1 - out2).abs()
        # ARRAY_START embedding should NOT change — it uses the learned token embedding
        assert diff[is_arr].sum() == 0

    def test_loss_includes_array_length_positions(self):
        """Model loss is computed at ARRAY_START positions."""
        data = [
            {"items": [1, 2], "val": ScaledNumeric(0.5)},
            {"items": [3], "val": ScaledNumeric(-0.1)},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(
            use_continuous_head=True,
            num_mixture_components=3,
            d_model=32,
            n_heads=2,
            n_layers=2,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        model.train()

        instances = [tokenizer.tokenize(obj) for obj in data]
        collator = OrigamiDataCollator(tokenizer, model_array_lengths=True)
        batch = collator(instances)

        # Verify ARRAY_START is in numeric_mask
        is_arr = batch.input_ids == 4
        assert (batch.numeric_mask & is_arr).any()

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
        assert output.loss.isfinite()


class TestSampleInteger:
    """Tests for discretized logistic integer sampling."""

    def test_basic_sampling_returns_integers(self):
        """sample_integer returns integer values in valid range."""
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=3))
        weights = torch.tensor([[[0.5, 0.3, 0.2]]])
        means = torch.tensor([[[0.3, 0.5, 0.7]]])
        log_vars = torch.tensor([[[-2.0, -2.0, -2.0]]])
        max_values = torch.tensor([[5.0]])

        torch.manual_seed(42)
        sampled = head.sample_integer(weights, means, log_vars, max_values)
        assert sampled.shape == (1, 1)
        assert sampled.dtype == torch.int64
        assert 0 <= sampled.item() <= 5

    def test_min_values_respected(self):
        """Samples are never below min_values."""
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=1))
        # Component centered at 0 — without min_values, would often sample 0
        weights = torch.tensor([[[1.0]]])
        means = torch.tensor([[[0.0]]])
        log_vars = torch.tensor([[[-2.0]]])
        max_values = torch.tensor([[5.0]])
        min_values = torch.tensor([[2]])

        samples = []
        for seed in range(50):
            torch.manual_seed(seed)
            s = head.sample_integer(weights, means, log_vars, max_values, min_values)
            samples.append(s.item())
        assert all(s >= 2 for s in samples)

    def test_max_values_respected(self):
        """Samples are never above max_values."""
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=1))
        # Component centered at 1.0 — would want to sample high
        weights = torch.tensor([[[1.0]]])
        means = torch.tensor([[[1.0]]])
        log_vars = torch.tensor([[[-2.0]]])
        max_values = torch.tensor([[3.0]])

        samples = []
        for seed in range(50):
            torch.manual_seed(seed)
            s = head.sample_integer(weights, means, log_vars, max_values)
            samples.append(s.item())
        assert all(s <= 3 for s in samples)

    def test_sharp_component_at_zero_samples_zero(self):
        """Sharp component at 0 should mostly sample 0."""
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=1))
        weights = torch.tensor([[[1.0]]])
        means = torch.tensor([[[0.0]]])
        log_vars = torch.tensor([[[-6.0]]])  # very sharp
        max_values = torch.tensor([[5.0]])

        samples = []
        for seed in range(100):
            torch.manual_seed(seed)
            s = head.sample_integer(weights, means, log_vars, max_values)
            samples.append(s.item())
        # Should be overwhelmingly 0 due to boundary absorption
        assert sum(1 for s in samples if s == 0) > 80

    def test_sharp_component_at_one_samples_max(self):
        """Sharp component at 1.0 should mostly sample max_value."""
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=1))
        weights = torch.tensor([[[1.0]]])
        means = torch.tensor([[[1.0]]])
        log_vars = torch.tensor([[[-6.0]]])  # very sharp
        max_values = torch.tensor([[5.0]])

        samples = []
        for seed in range(100):
            torch.manual_seed(seed)
            s = head.sample_integer(weights, means, log_vars, max_values)
            samples.append(s.item())
        assert sum(1 for s in samples if s == 5) > 80

    def test_batch_with_different_max_values(self):
        """Different max_values per position are handled correctly."""
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=1))
        weights = torch.tensor([[[1.0]], [[1.0]]])  # batch=2
        means = torch.tensor([[[0.5]], [[0.5]]])
        log_vars = torch.tensor([[[-2.0]], [[-2.0]]])
        max_values = torch.tensor([[3.0], [10.0]])

        samples_0, samples_1 = [], []
        for seed in range(50):
            torch.manual_seed(seed)
            s = head.sample_integer(weights, means, log_vars, max_values)
            samples_0.append(s[0, 0].item())
            samples_1.append(s[1, 0].item())
        assert all(s <= 3 for s in samples_0)
        assert all(s <= 10 for s in samples_1)

    def test_boundary_absorption_matches_training(self):
        """Verify that sample_integer bin probabilities match the training loss.

        For a single component, the probability of sampling bin k should match
        the bin probability computed by _nll_loss_discretized_logistic.
        """
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=1))
        weights = torch.tensor([[[1.0]]])
        means = torch.tensor([[[0.3]]])
        log_vars = torch.tensor([[[-1.0]]])
        max_items = 5

        # Collect empirical sampling distribution
        from collections import Counter

        counts = Counter()
        n_samples = 5000
        for seed in range(n_samples):
            torch.manual_seed(seed)
            s = head.sample_integer(weights, means, log_vars, torch.tensor([[float(max_items)]]))
            counts[s.item()] += 1

        # Compute theoretical bin probabilities from logistic CDF
        mu = 0.3
        scale = torch.exp(0.5 * torch.tensor(-1.0)).item()
        step = 1.0 / max_items

        theoretical_probs = {}
        for k in range(max_items + 1):
            t = k / max_items
            half_step = step / 2
            if k == 0:
                p = torch.sigmoid(torch.tensor((t + half_step - mu) / scale)).item()
            elif k == max_items:
                p = 1.0 - torch.sigmoid(torch.tensor((t - half_step - mu) / scale)).item()
            else:
                cdf_plus = torch.sigmoid(torch.tensor((t + half_step - mu) / scale)).item()
                cdf_minus = torch.sigmoid(torch.tensor((t - half_step - mu) / scale)).item()
                p = cdf_plus - cdf_minus
            theoretical_probs[k] = p

        # Normalize
        total = sum(theoretical_probs.values())
        for k in theoretical_probs:
            theoretical_probs[k] /= total

        # Compare empirical vs theoretical (allow 3% tolerance)
        for k in range(max_items + 1):
            empirical = counts.get(k, 0) / n_samples
            theoretical = theoretical_probs[k]
            assert abs(empirical - theoretical) < 0.03, (
                f"Bin {k}: empirical={empirical:.3f}, theoretical={theoretical:.3f}"
            )
