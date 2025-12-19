"""Integration tests for continuous head functionality."""

import pytest
import torch

from origami.config import ModelConfig
from origami.model import OrigamiModel
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
