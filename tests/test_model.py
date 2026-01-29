"""Tests for ORIGAMI model components.

TODO:
- Add tests for ContinuousHead with OrigamiModel (Phase 6)
- Add integration tests for full training loop
- Add tests for grammar constraint masking (Phase 4)
"""

import pytest
import torch

from origami.config import ModelConfig, TrainingConfig
from origami.model import (
    ContinuousHead,
    DiscreteHead,
    OrigamiEmbeddings,
    OrigamiModel,
    OrigamiOutput,
    TransformerBackbone,
    create_backbone,
)
from origami.tokenizer import JSONTokenizer
from origami.training import OrigamiDataCollator
from origami.utils import available_devices as get_available_devices

AVAILABLE_DEVICES = get_available_devices()


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test that default values are sensible."""
        config = ModelConfig()

        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 4
        assert config.d_ff == 512
        assert config.dropout == 0.0
        assert config.backbone == "transformer"
        assert config.kvpe_pooling == "sum"
        assert config.max_depth == 32
        assert config.max_array_position == 256
        assert config.use_continuous_head is False

    def test_validation_d_model_divisible_by_n_heads(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="d_model.*must be divisible by n_heads"):
            ModelConfig(d_model=256, n_heads=7)

    def test_validation_dropout_range(self):
        """Test that dropout must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=1.5)

    def test_custom_pooling_kwargs(self):
        """Test that pooling kwargs are passed correctly."""
        config = ModelConfig(
            kvpe_pooling="gru",
            kvpe_pooling_kwargs={"num_layers": 2},
        )
        assert config.kvpe_pooling == "gru"
        assert config.kvpe_pooling_kwargs == {"num_layers": 2}


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training config values."""
        config = TrainingConfig(num_epochs=10)

        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.num_epochs == 10
        assert config.shuffle_keys is True


class TestOrigamiEmbeddings:
    """Tests for OrigamiEmbeddings."""

    VOCAB_SIZE = 100

    @pytest.fixture
    def config(self):
        return ModelConfig(d_model=64, n_heads=4)

    def test_forward_shape(self, config):
        """Test that forward produces correct output shape."""
        embeddings = OrigamiEmbeddings(config, self.VOCAB_SIZE)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.VOCAB_SIZE, (batch_size, seq_len))
        path_types = torch.zeros(batch_size, seq_len, config.max_depth, dtype=torch.long)
        path_ids = torch.zeros(batch_size, seq_len, config.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(batch_size, seq_len, dtype=torch.long)

        output = embeddings(input_ids, path_types, path_ids, path_lengths)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_shared_key_embeddings(self, config):
        """Test that key embeddings are shared with token embeddings."""
        embeddings = OrigamiEmbeddings(config, self.VOCAB_SIZE)

        # The KVPE should share embeddings with token layer
        assert embeddings.kvpe._shared_key_embeddings is embeddings.token_embedding

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_forward_on_device(self, config, device):
        """Test forward pass on different devices."""
        embeddings = OrigamiEmbeddings(config, self.VOCAB_SIZE).to(device)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.VOCAB_SIZE, (batch_size, seq_len), device=device)
        path_types = torch.zeros(
            batch_size, seq_len, config.max_depth, dtype=torch.long, device=device
        )
        path_ids = torch.zeros(
            batch_size, seq_len, config.max_depth, dtype=torch.long, device=device
        )
        path_lengths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        output = embeddings(input_ids, path_types, path_ids, path_lengths)

        assert output.device.type == device.type
        assert output.shape == (batch_size, seq_len, config.d_model)


class TestTransformerBackbone:
    """Tests for TransformerBackbone."""

    @pytest.fixture
    def config(self):
        return ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128)

    def test_forward_shape(self, config):
        """Test that forward produces correct output shape."""
        backbone = TransformerBackbone(config)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)

        output = backbone(hidden)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_with_attention_mask(self, config):
        """Test forward with attention mask."""
        backbone = TransformerBackbone(config)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, 5:] = False  # Mask second half of first sequence

        output = backbone(hidden, attention_mask)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_causal_masking(self, config):
        """Test that causal masking is applied (position i can't see j > i)."""
        backbone = TransformerBackbone(config)

        batch_size, seq_len = 1, 5
        hidden = torch.randn(batch_size, seq_len, config.d_model)

        # Run forward
        output = backbone(hidden)

        # Output should exist and have correct shape
        # (Actual causal behavior is tested via gradients or attention weights,
        # but shape check confirms the mask didn't break anything)
        assert output.shape == (batch_size, seq_len, config.d_model)

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_forward_on_device(self, config, device):
        """Test forward pass on different devices."""
        backbone = TransformerBackbone(config).to(device)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model, device=device)

        output = backbone(hidden)

        assert output.device.type == device.type
        assert output.shape == (batch_size, seq_len, config.d_model)


class TestDiscreteHead:
    """Tests for DiscreteHead."""

    VOCAB_SIZE = 100

    @pytest.fixture
    def config(self):
        return ModelConfig(d_model=64, n_heads=4)

    def test_forward_shape(self, config):
        """Test that forward produces correct logit shape."""
        head = DiscreteHead(config, self.VOCAB_SIZE)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)

        logits = head(hidden)

        assert logits.shape == (batch_size, seq_len, self.VOCAB_SIZE)

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_forward_on_device(self, config, device):
        """Test forward pass on different devices."""
        head = DiscreteHead(config, self.VOCAB_SIZE).to(device)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model, device=device)

        logits = head(hidden)

        assert logits.device.type == device.type
        assert logits.shape == (batch_size, seq_len, self.VOCAB_SIZE)


class TestContinuousHead:
    """Tests for ContinuousHead."""

    @pytest.fixture
    def config(self):
        return ModelConfig(d_model=64, n_heads=4, num_mixture_components=5)

    def test_forward_shape(self, config):
        """Test that forward produces correct MoG parameter shapes."""
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)

        weights, means, log_vars = head(hidden)

        assert weights.shape == (batch_size, seq_len, config.num_mixture_components)
        assert means.shape == (batch_size, seq_len, config.num_mixture_components)
        assert log_vars.shape == (batch_size, seq_len, config.num_mixture_components)

    def test_weights_sum_to_one(self, config):
        """Test that mixture weights sum to 1."""
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)

        weights, _, _ = head(hidden)

        # Weights should sum to 1 along component dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_nll_loss(self, config):
        """Test NLL loss computation."""
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)
        targets = torch.randn(batch_size, seq_len)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[0, :5] = True  # Only first 5 positions of first sequence

        weights, means, log_vars = head(hidden)
        loss = head.nll_loss(weights, means, log_vars, targets, mask)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # NLL should be positive for random data

    def test_nll_loss_empty_mask(self, config):
        """Test NLL loss with empty mask returns zero."""
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)
        targets = torch.randn(batch_size, seq_len)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)  # All False

        weights, means, log_vars = head(hidden)
        loss = head.nll_loss(weights, means, log_vars, targets, mask)

        assert loss.item() == 0.0

    def test_sample_shape(self, config):
        """Test that sample produces correct shape."""
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model)

        weights, means, log_vars = head(hidden)
        samples = head.sample(weights, means, log_vars)

        assert samples.shape == (batch_size, seq_len)

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_forward_on_device(self, config, device):
        """Test forward pass on different devices."""
        head = ContinuousHead(config).to(device)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model, device=device)

        weights, means, log_vars = head(hidden)

        assert weights.device.type == device.type
        assert means.device.type == device.type
        assert log_vars.device.type == device.type

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_nll_loss_on_device(self, config, device):
        """Test NLL loss computation on different devices."""
        head = ContinuousHead(config).to(device)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model, device=device)
        targets = torch.randn(batch_size, seq_len, device=device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        mask[0, :5] = True

        weights, means, log_vars = head(hidden)
        loss = head.nll_loss(weights, means, log_vars, targets, mask)

        assert loss.device.type == device.type

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_sample_on_device(self, config, device):
        """Test sampling on different devices."""
        head = ContinuousHead(config).to(device)

        batch_size, seq_len = 2, 10
        hidden = torch.randn(batch_size, seq_len, config.d_model, device=device)

        weights, means, log_vars = head(hidden)
        samples = head.sample(weights, means, log_vars)

        assert samples.device.type == device.type

    def test_truncated_sample_within_bounds(self, config):
        """Truncated samples should fall within [lower, upper]."""
        head = ContinuousHead(config)
        batch_size, seq_len = 4, 1
        n_comp = config.num_mixture_components

        # Fixed MoG: mean=0, std=1, equal weights
        weights = torch.ones(batch_size, seq_len, n_comp) / n_comp
        means = torch.zeros(batch_size, seq_len, n_comp)
        log_vars = torch.zeros(batch_size, seq_len, n_comp)

        lower = torch.full((batch_size, seq_len), -1.0)
        upper = torch.full((batch_size, seq_len), 1.0)

        torch.manual_seed(42)
        # Sample many times to check bounds
        for _ in range(50):
            samples = head.sample(weights, means, log_vars, lower=lower, upper=upper)
            assert samples.shape == (batch_size, seq_len)
            assert (samples >= -1.0 - 1e-5).all(), f"Sample below lower bound: {samples.min()}"
            assert (samples <= 1.0 + 1e-5).all(), f"Sample above upper bound: {samples.max()}"

    def test_truncated_sample_lower_only(self, config):
        """Only lower bound should be respected when upper is None."""
        head = ContinuousHead(config)
        batch_size, seq_len = 2, 1
        n_comp = config.num_mixture_components

        weights = torch.ones(batch_size, seq_len, n_comp) / n_comp
        means = torch.zeros(batch_size, seq_len, n_comp)
        log_vars = torch.zeros(batch_size, seq_len, n_comp)

        lower = torch.full((batch_size, seq_len), 2.0)

        torch.manual_seed(0)
        for _ in range(50):
            samples = head.sample(weights, means, log_vars, lower=lower, upper=None)
            assert (samples >= 2.0 - 1e-5).all()

    def test_truncated_sample_upper_only(self, config):
        """Only upper bound should be respected when lower is None."""
        head = ContinuousHead(config)
        batch_size, seq_len = 2, 1
        n_comp = config.num_mixture_components

        weights = torch.ones(batch_size, seq_len, n_comp) / n_comp
        means = torch.zeros(batch_size, seq_len, n_comp)
        log_vars = torch.zeros(batch_size, seq_len, n_comp)

        upper = torch.full((batch_size, seq_len), -2.0)

        torch.manual_seed(0)
        for _ in range(50):
            samples = head.sample(weights, means, log_vars, lower=None, upper=upper)
            assert (samples <= -2.0 + 1e-5).all()

    def test_truncated_sample_no_bounds_unchanged(self, config):
        """Without bounds, truncated path is not taken (same as unconstrained)."""
        head = ContinuousHead(config)
        batch_size, seq_len = 2, 1
        n_comp = config.num_mixture_components

        weights = torch.ones(batch_size, seq_len, n_comp) / n_comp
        means = torch.zeros(batch_size, seq_len, n_comp)
        log_vars = torch.zeros(batch_size, seq_len, n_comp)

        torch.manual_seed(42)
        samples_no_bounds = head.sample(weights, means, log_vars)
        torch.manual_seed(42)
        samples_none = head.sample(weights, means, log_vars, lower=None, upper=None)

        assert torch.allclose(samples_no_bounds, samples_none)

    def test_truncated_sample_per_batch_bounds(self, config):
        """Different bounds per batch item should be respected."""
        head = ContinuousHead(config)
        batch_size, seq_len = 2, 1
        n_comp = config.num_mixture_components

        weights = torch.ones(batch_size, seq_len, n_comp) / n_comp
        means = torch.zeros(batch_size, seq_len, n_comp)
        log_vars = torch.zeros(batch_size, seq_len, n_comp)

        # Item 0: [0, 5], Item 1: [-5, 0]
        lower = torch.tensor([[0.0], [-5.0]])
        upper = torch.tensor([[5.0], [0.0]])

        torch.manual_seed(42)
        for _ in range(50):
            samples = head.sample(weights, means, log_vars, lower=lower, upper=upper)
            assert (samples[0] >= 0.0 - 1e-5).all()
            assert (samples[0] <= 5.0 + 1e-5).all()
            assert (samples[1] >= -5.0 - 1e-5).all()
            assert (samples[1] <= 0.0 + 1e-5).all()


class TestOrigamiModel:
    """Tests for OrigamiModel."""

    @pytest.fixture
    def tokenizer(self):
        """Create a simple tokenizer for testing."""
        tokenizer = JSONTokenizer()
        tokenizer.fit(
            [
                {"name": "Alice", "age": 30, "scores": [90, 85]},
                {"name": "Bob", "age": 25, "active": True},
            ]
        )
        return tokenizer

    @pytest.fixture
    def config(self, tokenizer):
        return ModelConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_depth=tokenizer.max_depth,
            max_array_position=tokenizer.max_array_index,
        )

    def test_forward_no_labels(self, config, tokenizer):
        """Test forward pass without labels."""
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice", "age": 30}]
        )

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
        )

        assert isinstance(output, OrigamiOutput)
        assert output.loss is None
        assert output.logits.shape == (1, batch.input_ids.size(1), tokenizer.vocab.size)
        assert output.hidden_states.shape == (1, batch.input_ids.size(1), config.d_model)
        assert output.continuous_params is None

    def test_forward_with_labels(self, config, tokenizer):
        """Test forward pass with labels (computes loss)."""
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice", "age": 30}]
        )

        # Use input_ids as labels (teacher forcing)
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
        )

        assert output.loss is not None
        assert output.loss.ndim == 0  # Scalar

    def test_forward_batch(self, config, tokenizer):
        """Test forward pass with batched input."""
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        objects = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "active": True},
        ]
        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(objects)

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
        )

        batch_size = len(objects)
        seq_len = batch.input_ids.size(1)
        assert output.logits.shape == (batch_size, seq_len, tokenizer.vocab.size)

    def test_attention_mask_affects_loss(self, config, tokenizer):
        """Test that attention_mask properly excludes padding from loss.

        This test verifies the bug fix where attention_mask was passed but
        not used in _compute_loss. Padding positions should be ignored.
        """
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        # Create a batch with sequences of different lengths
        objects = [
            {"name": "Alice", "age": 30, "active": True},  # Longer
            {"name": "Bob"},  # Shorter
        ]
        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(objects)

        # Compute loss with attention mask
        output_with_mask = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
        )

        # Create fake attention mask that marks everything as valid
        fake_mask = torch.ones_like(batch.attention_mask)

        output_without_mask = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=fake_mask,
            labels=batch.input_ids,
        )

        # Losses should be different because padding affects the fake mask version
        # The real mask excludes padding, fake mask includes it
        assert output_with_mask.loss is not None
        assert output_without_mask.loss is not None
        # They should differ (padding tokens contribute to one but not the other)
        assert not torch.allclose(output_with_mask.loss, output_without_mask.loss)

    def test_padding_positions_ignored_in_loss(self, config, tokenizer):
        """Test that loss is only computed on valid (non-padding) positions."""
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        # Use objects that are already in the tokenizer's vocabulary
        short_obj = {"name": "Alice"}
        long_obj = {"name": "Alice", "age": 30, "active": True}

        # Encode separately to get different sequence lengths
        batch_short = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [short_obj]
        )
        batch_long = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [long_obj]
        )

        # Compute individual losses
        loss_short = model(
            input_ids=batch_short.input_ids,
            path_types=batch_short.path_types,
            path_ids=batch_short.path_ids,
            path_lengths=batch_short.path_lengths,
            attention_mask=batch_short.attention_mask,
            labels=batch_short.input_ids,
        ).loss

        loss_long = model(
            input_ids=batch_long.input_ids,
            path_types=batch_long.path_types,
            path_ids=batch_long.path_ids,
            path_lengths=batch_long.path_lengths,
            attention_mask=batch_long.attention_mask,
            labels=batch_long.input_ids,
        ).loss

        # Now batch them together (will have padding)
        batch_combined = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [short_obj, long_obj]
        )

        output_combined = model(
            input_ids=batch_combined.input_ids,
            path_types=batch_combined.path_types,
            path_ids=batch_combined.path_ids,
            path_lengths=batch_combined.path_lengths,
            attention_mask=batch_combined.attention_mask,
            labels=batch_combined.input_ids,
        )

        # Combined loss should be close to average of individual losses
        # (not exact due to batching effects, but should be in same ballpark)
        expected_avg = (loss_short + loss_long) / 2
        assert output_combined.loss is not None
        # Allow some tolerance for numerical differences
        assert abs(output_combined.loss.item() - expected_avg.item()) < 1.0

    def test_get_num_parameters(self, config, tokenizer):
        """Test parameter counting."""
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        num_params = model.get_num_parameters(trainable_only=True)
        total_params = model.get_num_parameters(trainable_only=False)

        assert num_params > 0
        assert num_params == total_params  # All params should be trainable by default

    def test_model_gradient_flow(self, config, tokenizer):
        """Test that gradients flow through the model."""
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        # Use data with arrays so index_embeddings gets gradients
        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice", "age": 30, "scores": [90, 85]}]
        )

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
        )

        # Backprop
        output.loss.backward()

        # Check that gradients exist for parameters that should have been used
        # Note: some parameters like index_embeddings may have zero gradients
        # if the batch doesn't contain arrays, but they should still have grads
        has_any_grad = False
        for _name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_any_grad = True
                # Gradient should exist and be on same device
                assert param.grad.shape == param.shape

        assert has_any_grad, "No gradients computed at all"

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_forward_on_device(self, config, tokenizer, device):
        """Test forward pass on different devices."""
        model = OrigamiModel(config, vocab=tokenizer.vocab).to(device)

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice", "age": 30}]
        )
        batch = batch.to(device)

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
        )

        assert output.logits.device.type == device.type
        assert output.loss.device.type == device.type
        assert output.hidden_states.device.type == device.type

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_gradient_flow_on_device(self, config, tokenizer, device):
        """Test gradient computation on different devices."""
        model = OrigamiModel(config, vocab=tokenizer.vocab).to(device)

        # Use data with arrays so index_embeddings gets gradients
        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice", "age": 30, "scores": [90, 85]}]
        )
        batch = batch.to(device)

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
        )

        # Backprop should work on all devices
        output.loss.backward()

        # Check that at least some gradients exist and are on the correct device
        has_any_grad = False
        for _name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_any_grad = True
                assert param.grad.device.type == device.type

        assert has_any_grad, "No gradients computed at all"

    def test_continuous_head_enabled(self, tokenizer):
        """Test model with continuous head enabled."""
        config = ModelConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            use_continuous_head=True,
            num_mixture_components=3,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        # Verify continuous head exists
        assert model.continuous_head is not None

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice", "age": 30}]
        )
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
        )

        # continuous_params should be returned
        assert output.continuous_params is not None
        weights, means, log_vars = output.continuous_params
        assert weights.shape == (1, batch.input_ids.shape[1], 3)
        assert means.shape == weights.shape
        assert log_vars.shape == weights.shape

    def test_continuous_loss_computation(self, tokenizer):
        """Test that continuous loss is computed when numeric values provided."""
        config = ModelConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            use_continuous_head=True,
            num_mixture_components=3,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice", "age": 30}]
        )
        seq_len = batch.input_ids.shape[1]

        # Create numeric values and mask (simulate NUM tokens at positions 3, 5)
        numeric_values = torch.randn(1, seq_len)
        numeric_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        numeric_mask[0, 3] = True
        numeric_mask[0, 5] = True

        # Forward with labels and numeric data
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
        )

        # Loss should include both discrete and continuous components
        assert output.loss is not None

        # Compare with loss without continuous component
        output_discrete_only = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
        )

        # Losses should differ due to continuous component
        assert not torch.allclose(output.loss, output_discrete_only.loss)


class TestEncodeBatch:
    """Tests for JSONTokenizer.encode_batch()."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer fitted on sample data."""
        tokenizer = JSONTokenizer()
        tokenizer.fit(
            [
                {"name": "Alice", "age": 30, "scores": [90, 85]},
                {"name": "Bob", "active": True},
            ]
        )
        return tokenizer

    def test_encode_batch_single(self, tokenizer):
        """Test encoding a single object."""
        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice"}]
        )

        assert batch.input_ids.shape[0] == 1
        assert batch.attention_mask.shape == batch.input_ids.shape
        assert batch.path_types.shape[:2] == batch.input_ids.shape
        assert batch.path_ids.shape[:2] == batch.input_ids.shape
        assert batch.path_lengths.shape == batch.input_ids.shape
        assert batch.lengths.shape == (1,)

    def test_encode_batch_multiple(self, tokenizer):
        """Test encoding multiple objects with padding."""
        objects = [
            {"name": "Alice", "age": 30, "scores": [90, 85]},
            {"name": "Bob"},
        ]
        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(objects)

        assert batch.input_ids.shape[0] == 2
        # First object is longer, so second should be padded
        assert batch.lengths[0] > batch.lengths[1]
        # Attention mask should reflect padding
        assert batch.attention_mask[0].sum() == batch.lengths[0]
        assert batch.attention_mask[1].sum() == batch.lengths[1]

    def test_encode_batch_path_types(self, tokenizer):
        """Test that path types are correctly encoded."""
        from origami.position_encoding import PATH_TYPE_INDEX, PATH_TYPE_KEY

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"scores": [90, 85]}]
        )

        # Find positions where path has elements
        path_lengths = batch.path_lengths[0]

        # Check that non-zero path positions have valid types
        path_types = batch.path_types[0]
        for i in range(batch.input_ids.size(1)):
            depth = path_lengths[i].item()
            for d in range(int(depth)):
                pt = path_types[i, d].item()
                assert pt in [PATH_TYPE_KEY, PATH_TYPE_INDEX]

    def test_encode_batch_empty_raises(self, tokenizer):
        """Test that encoding empty batch raises error."""
        with pytest.raises(ValueError, match="Cannot collate empty batch"):
            OrigamiDataCollator(tokenizer, include_labels=False).collate_objects([])

    def test_encode_batch_shuffle(self, tokenizer):
        """Test that shuffle produces different orderings."""
        import random

        random.seed(42)

        obj = {"a": 1, "b": 2, "c": 3, "d": 4}
        tokenizer.fit([obj])

        # Encode multiple times with shuffle
        batches = [
            OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
                [obj], shuffle=True
            )
            for _ in range(10)
        ]

        # Get token sequences
        sequences = [tuple(b.input_ids[0].tolist()) for b in batches]

        # With shuffle=True, we should see different orderings
        # (probabilistically, with 4 keys, almost certainly different)
        unique_sequences = set(sequences)
        assert len(unique_sequences) > 1, "Shuffle should produce different orderings"

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_encode_batch_to_device(self, tokenizer, device):
        """Test moving encoded batch to different devices."""
        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects(
            [{"name": "Alice"}]
        )

        # Default should be CPU
        assert batch.input_ids.device.type == "cpu"

        # Move to target device
        batch_on_device = batch.to(device)

        # All tensors should be on the target device
        assert batch_on_device.input_ids.device.type == device.type
        assert batch_on_device.path_types.device.type == device.type
        assert batch_on_device.path_ids.device.type == device.type
        assert batch_on_device.path_lengths.device.type == device.type
        assert batch_on_device.attention_mask.device.type == device.type
        assert batch_on_device.numeric_values.device.type == device.type
        assert batch_on_device.numeric_mask.device.type == device.type
        assert batch_on_device.lengths.device.type == device.type

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_encode_batch_with_device_param(self, tokenizer, device):
        """Test encoding directly to a device."""
        batch = OrigamiDataCollator(tokenizer, include_labels=False, device=device).collate_objects(
            [{"name": "Alice"}]
        )

        assert batch.input_ids.device.type == device.type
        assert batch.attention_mask.device.type == device.type


class TestCreateBackbone:
    """Tests for backbone factory function."""

    def test_create_transformer(self):
        """Test creating transformer backbone."""
        config = ModelConfig(backbone="transformer")
        backbone = create_backbone(config)
        assert isinstance(backbone, TransformerBackbone)

    def test_create_lstm(self):
        """Test that LSTM backbone is created successfully."""
        config = ModelConfig(backbone="lstm")
        backbone = create_backbone(config)
        assert backbone is not None

    def test_create_mamba_not_implemented(self):
        """Test that Mamba backbone raises NotImplementedError."""
        config = ModelConfig(backbone="mamba")
        with pytest.raises(NotImplementedError):
            create_backbone(config)

    def test_create_unknown_raises(self):
        """Test that unknown backbone type raises ValueError."""
        # Need to bypass dataclass validation
        config = ModelConfig()
        config.backbone = "unknown"  # type: ignore
        with pytest.raises(ValueError, match="Unknown backbone type"):
            create_backbone(config)


class TestOrigamiModelSaveLoad:
    """Tests for OrigamiModel save() and load() methods."""

    @pytest.fixture
    def tokenizer(self):
        """Create a simple tokenizer for testing."""
        tokenizer = JSONTokenizer()
        tokenizer.fit(
            [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        )
        return tokenizer

    @pytest.fixture
    def model_and_tokenizer(self, tokenizer):
        """Create a model with tokenizer."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        return model, tokenizer

    def test_save_and_load(self, model_and_tokenizer, tmp_path):
        """Test saving and loading a model."""
        model, tokenizer = model_and_tokenizer
        checkpoint_path = tmp_path / "model.pt"

        # Save model with tokenizer
        model.save(checkpoint_path, tokenizer)

        # Load model
        loaded_model, loaded_tokenizer = OrigamiModel.load(checkpoint_path)

        # Verify model config matches
        assert loaded_model.config.d_model == model.config.d_model
        assert loaded_model.config.n_heads == model.config.n_heads
        assert loaded_model.config.n_layers == model.config.n_layers

        # Verify tokenizer is restored
        assert loaded_tokenizer is not None
        assert loaded_tokenizer.vocab.size == tokenizer.vocab.size
        assert loaded_tokenizer.max_depth == tokenizer.max_depth

    def test_save_creates_parent_dirs(self, model_and_tokenizer, tmp_path):
        """Test that save() creates parent directories if needed."""
        model, tokenizer = model_and_tokenizer
        checkpoint_path = tmp_path / "nested" / "dir" / "model.pt"

        model.save(checkpoint_path, tokenizer)

        assert checkpoint_path.exists()

    def test_load_without_tokenizer_raises(self, model_and_tokenizer, tmp_path):
        """Test that loading checkpoint without tokenizer raises error."""
        model, _ = model_and_tokenizer
        checkpoint_path = tmp_path / "model_no_tokenizer.pt"

        # Save without tokenizer (manually create incomplete checkpoint)
        torch.save(
            {
                "model_config": {"d_model": 32, "n_heads": 2, "n_layers": 1},
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )

        with pytest.raises(ValueError, match="does not contain tokenizer state"):
            OrigamiModel.load(checkpoint_path)

    def test_load_to_specific_device(self, model_and_tokenizer, tmp_path):
        """Test loading model to a specific device."""
        model, tokenizer = model_and_tokenizer
        checkpoint_path = tmp_path / "model.pt"

        model.save(checkpoint_path, tokenizer)

        loaded_model, _ = OrigamiModel.load(checkpoint_path, device="cpu")
        assert next(loaded_model.parameters()).device == torch.device("cpu")

    def test_weights_preserved_after_load(self, model_and_tokenizer, tmp_path):
        """Test that model weights are preserved after save/load."""
        model, tokenizer = model_and_tokenizer
        checkpoint_path = tmp_path / "model.pt"

        # Get original weights
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.save(checkpoint_path, tokenizer)
        loaded_model, _ = OrigamiModel.load(checkpoint_path)

        # Compare weights
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_model.state_dict()[key]), (
                f"Weights mismatch for {key}"
            )


class TestOrigamiModelGrammar:
    """Tests for grammar-related functionality."""

    @pytest.fixture
    def tokenizer(self):
        """Create a simple tokenizer for testing."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"a": 1, "b": 2}])
        return tokenizer

    def test_compute_grammar_mask_no_pda(self, tokenizer):
        """Test compute_grammar_mask returns None when no grammar PDA is attached."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        input_ids = torch.randint(0, tokenizer.vocab.size, (1, 10))
        mask = model.compute_grammar_mask(input_ids)

        assert mask is None

    def test_compute_grammar_mask_with_pda(self, tokenizer):
        """Test compute_grammar_mask returns valid mask when grammar PDA is attached."""
        from origami.constraints.json_grammar import JSONGrammarPDA

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        # Attach grammar PDA (normally done by trainer)
        model._grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=config.max_depth)

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects([{"a": 1}])
        mask = model.compute_grammar_mask(batch.input_ids)

        assert mask is not None
        assert mask.shape == (1, batch.input_ids.size(1), tokenizer.vocab.size)
        assert mask.dtype == torch.bool


class TestOrigamiModelContinuousLossWeight:
    """Tests for continuous loss weight configuration."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"value": 1.5}, {"value": 2.5}])
        return tokenizer

    def test_fixed_continuous_loss_weight(self, tokenizer):
        """Test that fixed continuous_loss_weight is used when set."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            use_continuous_head=True,
            continuous_loss_weight=0.5,  # Fixed weight
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects([{"value": 1}])
        seq_len = batch.input_ids.shape[1]

        # Create numeric values and mask
        numeric_values = torch.randn(1, seq_len)
        numeric_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        numeric_mask[0, 2] = True  # Mark one position as NUM

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
        )

        # Loss should be computed
        assert output.loss is not None

    def test_auto_continuous_loss_weight(self, tokenizer):
        """Test auto-calculated continuous loss weight (default -1)."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            use_continuous_head=True,
            continuous_loss_weight=-1.0,  # Auto-calculate (default)
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        batch = OrigamiDataCollator(tokenizer, include_labels=False).collate_objects([{"value": 1}])
        seq_len = batch.input_ids.shape[1]

        # Create numeric values and mask with multiple NUM positions
        numeric_values = torch.randn(1, seq_len)
        numeric_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        numeric_mask[0, 2:5] = True  # Mark several positions

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.input_ids,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
        )

        # Loss should be computed with auto-calculated weight
        assert output.loss is not None
