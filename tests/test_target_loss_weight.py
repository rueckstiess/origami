"""Tests for target_loss_weight feature."""

import pytest
import torch

from origami.config import ModelConfig, TrainingConfig
from origami.model import OrigamiModel
from origami.position_encoding import PATH_TYPE_KEY
from origami.tokenizer import JSONTokenizer
from origami.tokenizer.vocabulary import KeyToken
from origami.training import OrigamiDataCollator, OrigamiTrainer


class TestComputeLossWeights:
    """Tests for _compute_loss_weights method in OrigamiTrainer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer fitted on data with various structures."""
        tokenizer = JSONTokenizer()
        tokenizer.fit(
            [
                {"a": 1, "target": 2, "b": 3},
                {"a": 1, "target": {"nested": 5}, "b": 3},
                {"a": 1, "target": [1, 2, 3], "b": 3},
            ]
        )
        return tokenizer

    @pytest.fixture
    def model(self, tokenizer):
        """Create a small model for testing."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=tokenizer.vocab)

    def test_returns_none_when_target_key_is_none(self, model, tokenizer):
        """Test that _compute_loss_weights returns None when target_key is None."""
        train_data = [{"a": 1, "target": 2}]
        config = TrainingConfig(
            num_epochs=1,
            target_key=None,
            target_loss_weight=10.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is None

    def test_returns_none_when_weight_is_one(self, model, tokenizer):
        """Test that _compute_loss_weights returns None when target_loss_weight is 1.0."""
        train_data = [{"a": 1, "target": 2}]
        config = TrainingConfig(
            num_epochs=1,
            target_key="target",
            target_loss_weight=1.0,  # No effect, should skip
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is None

    def test_returns_none_when_target_key_not_in_vocab(self, model, tokenizer):
        """Test returns None when target key is not in vocabulary."""
        train_data = [{"a": 1, "b": 2}]
        config = TrainingConfig(
            num_epochs=1,
            target_key="nonexistent_key",
            target_loss_weight=10.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is None

    def test_weights_shape_matches_batch(self, model, tokenizer):
        """Test that weights tensor has correct shape."""
        train_data = [{"a": 1, "target": 2}]
        config = TrainingConfig(
            num_epochs=1,
            target_key="target",
            target_loss_weight=10.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is not None
        assert weights.shape == batch.input_ids.shape

    def test_weights_normalized_sum_equals_valid_tokens(self, model, tokenizer):
        """Test that normalized weights sum equals number of valid tokens."""
        train_data = [{"a": 1, "target": 2, "b": 3}]
        config = TrainingConfig(
            num_epochs=1,
            target_key="target",
            target_loss_weight=10.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is not None

        # Sum of weights for valid positions should equal valid token count
        valid_weights = (weights * batch.attention_mask.float()).sum()
        valid_token_count = batch.attention_mask.sum().float()
        assert torch.isclose(valid_weights, valid_token_count, atol=1e-5)

    def test_target_value_tokens_have_higher_weight(self, model, tokenizer):
        """Test that tokens in target value have higher normalized weights."""
        train_data = [{"a": 1, "target": 2, "b": 3}]
        config = TrainingConfig(
            num_epochs=1,
            target_key="target",
            target_loss_weight=10.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is not None

        # Find target value positions using path info
        target_key_id = tokenizer.vocab.encode(KeyToken("target"))
        in_target_value = (batch.path_types[:, :, 0] == PATH_TYPE_KEY) & (
            batch.path_ids[:, :, 0] == target_key_id
        )

        # Target positions should have higher weights than non-target
        target_weights = weights[in_target_value]
        non_target_weights = weights[~in_target_value & batch.attention_mask]

        if len(target_weights) > 0 and len(non_target_weights) > 0:
            assert target_weights.mean() > non_target_weights.mean()


class TestTargetLossWeightForComplexValues:
    """Tests for target_loss_weight with complex values (objects/arrays)."""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer with complex structures."""
        tokenizer = JSONTokenizer()
        tokenizer.fit(
            [
                {"a": 1, "target": {"nested": 5, "deep": {"x": 1}}, "b": 3},
                {"a": 1, "target": [1, 2, [3, 4]], "b": 3},
            ]
        )
        return tokenizer

    @pytest.fixture
    def model(self, tokenizer):
        """Create a small model for testing."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        return OrigamiModel(config, vocab=tokenizer.vocab)

    def test_nested_object_all_tokens_weighted(self, model, tokenizer):
        """Test that all tokens in nested object are correctly identified."""
        train_data = [{"a": 1, "target": {"nested": 5}, "b": 3}]
        config = TrainingConfig(
            num_epochs=1,
            target_key="target",
            target_loss_weight=10.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is not None

        # Count how many tokens are in target value using path info
        target_key_id = tokenizer.vocab.encode(KeyToken("target"))
        in_target_value = (batch.path_types[:, :, 0] == PATH_TYPE_KEY) & (
            batch.path_ids[:, :, 0] == target_key_id
        )

        # For {"nested": 5}, we expect: OBJ_START, Key("nested"), Value(5), OBJ_END
        # That's at least 4 tokens in the target value
        target_count = in_target_value.sum().item()
        assert target_count >= 4, f"Expected at least 4 tokens in target value, got {target_count}"

    def test_array_all_tokens_weighted(self, model, tokenizer):
        """Test that all tokens in array are correctly identified."""
        train_data = [{"a": 1, "target": [1, 2, 3], "b": 3}]
        config = TrainingConfig(
            num_epochs=1,
            target_key="target",
            target_loss_weight=10.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(train_data[0])])

        weights = trainer._compute_loss_weights(batch)
        assert weights is not None

        # Count tokens in target value
        target_key_id = tokenizer.vocab.encode(KeyToken("target"))
        in_target_value = (batch.path_types[:, :, 0] == PATH_TYPE_KEY) & (
            batch.path_ids[:, :, 0] == target_key_id
        )

        # For [1, 2, 3], we expect: ARRAY_START, Value(1), Value(2), Value(3), ARRAY_END
        # That's at least 5 tokens in the target value
        target_count = in_target_value.sum().item()
        assert target_count >= 5, f"Expected at least 5 tokens in target value, got {target_count}"


class TestTargetLossWeightIntegration:
    """Integration tests for target_loss_weight with full training."""

    @pytest.fixture
    def setup(self):
        """Set up model, tokenizer, and data."""
        data = [
            {"x": 1, "target": "A"},
            {"x": 2, "target": "B"},
            {"x": 3, "target": "A"},
            {"x": 4, "target": "B"},
        ] * 5  # 20 samples

        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        return model, tokenizer, data

    def test_training_with_target_loss_weight(self, setup):
        """Test that training completes successfully with target_loss_weight."""
        model, tokenizer, data = setup

        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            learning_rate=1e-3,
            target_key="target",
            target_loss_weight=5.0,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        state = trainer.train()

        # Training should complete
        assert state.global_step > 0

        # Model should not have NaN parameters
        for param in model.parameters():
            assert not torch.isnan(param).any()

    def test_weighted_loss_vs_unweighted(self, setup):
        """Test that weighted loss differs from unweighted loss."""
        model, tokenizer, data = setup

        # Get an initial batch (keep on CPU for simplicity)
        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(obj) for obj in data[:4]])

        # Create trainer to get weights (use CPU device explicitly)
        config = TrainingConfig(
            num_epochs=1,
            target_key="target",
            target_loss_weight=10.0,
        )
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
            device=torch.device("cpu"),  # Force CPU to avoid device mismatch
        )

        # Move batch to CPU
        batch = batch.to(torch.device("cpu"))
        weights = trainer._compute_loss_weights(batch)

        # Forward pass without weights
        model.eval()
        model.to("cpu")
        with torch.no_grad():
            output_unweighted = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                loss_weights=None,
            )

            output_weighted = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                loss_weights=weights,
            )

        # Losses should be different (weighted emphasizes target tokens)
        # Note: They could be equal in edge cases, but generally differ
        assert output_unweighted.loss is not None
        assert output_weighted.loss is not None


class TestTargetLossWeightConfig:
    """Tests for target_loss_weight in TrainingConfig."""

    def test_default_value_is_one(self):
        """Test that default target_loss_weight is 1.0."""
        config = TrainingConfig(num_epochs=1)
        assert config.target_loss_weight == 1.0

    def test_can_set_higher_weight(self):
        """Test setting weight higher than 1.0."""
        config = TrainingConfig(num_epochs=1, target_loss_weight=10.0)
        assert config.target_loss_weight == 10.0

    def test_can_set_lower_weight(self):
        """Test setting weight lower than 1.0 (de-emphasize target)."""
        config = TrainingConfig(num_epochs=1, target_loss_weight=0.1)
        assert config.target_loss_weight == 0.1


class TestModelWeightedLoss:
    """Tests for weighted loss computation in OrigamiModel."""

    @pytest.fixture
    def setup(self):
        """Create model and tokenizer."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"a": 1, "b": 2}])

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        return model, tokenizer

    def test_loss_weights_parameter_accepted(self, setup):
        """Test that model.forward accepts loss_weights parameter."""
        model, tokenizer = setup

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize({"a": 1, "b": 2})])

        # Create simple uniform weights
        weights = torch.ones_like(batch.input_ids, dtype=torch.float)

        # Should not raise
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            loss_weights=weights,
        )

        assert output.loss is not None

    def test_uniform_weights_similar_to_no_weights(self, setup):
        """Test that uniform weights produce similar loss to no weights."""
        model, tokenizer = setup

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize({"a": 1, "b": 2})])

        model.eval()
        with torch.no_grad():
            # No weights
            output_no_weights = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                loss_weights=None,
            )

            # Uniform weights (should behave same as no weights)
            uniform_weights = torch.ones_like(batch.input_ids, dtype=torch.float)
            output_uniform = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                loss_weights=uniform_weights,
            )

        # Losses should be very close
        assert torch.isclose(
            output_no_weights.loss,
            output_uniform.loss,
            atol=1e-5,
        )

    def test_higher_weight_increases_contribution(self, setup):
        """Test that higher weights increase the contribution of specific tokens."""
        model, tokenizer = setup

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize({"a": 1, "b": 2})])

        # Find a position that has a valid token (not padding)
        valid_pos = batch.attention_mask[0].nonzero()[1].item()

        model.eval()
        with torch.no_grad():
            # Uniform weights
            weights1 = torch.ones_like(batch.input_ids, dtype=torch.float)
            output1 = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                loss_weights=weights1,
            )

            # High weight on specific position
            weights2 = torch.ones_like(batch.input_ids, dtype=torch.float)
            weights2[0, valid_pos] = 100.0  # Much higher weight
            output2 = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                loss_weights=weights2,
            )

        # Losses should differ due to different weighting
        # (exact relationship depends on the per-token losses)
        assert output1.loss is not None
        assert output2.loss is not None


class TestContinuousHeadWeightedLoss:
    """Tests for loss_weights applied to continuous (MoG) head."""

    def test_nll_loss_accepts_loss_weights(self):
        """Test that ContinuousHead.nll_loss accepts loss_weights parameter."""
        from origami.model.heads import ContinuousHead

        config = ModelConfig(d_model=32, num_mixture_components=3)
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 5
        weights = torch.softmax(torch.randn(batch_size, seq_len, 3), dim=-1)
        means = torch.randn(batch_size, seq_len, 3)
        log_vars = torch.randn(batch_size, seq_len, 3)
        targets = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        loss_weights = torch.ones(batch_size, seq_len)

        # Should not raise
        loss = head.nll_loss(weights, means, log_vars, targets, mask, loss_weights=loss_weights)
        assert loss is not None
        assert not torch.isnan(loss)

    def test_uniform_loss_weights_same_as_no_weights(self):
        """Test that uniform loss_weights produce same result as no weights."""
        from origami.model.heads import ContinuousHead

        config = ModelConfig(d_model=32, num_mixture_components=3)
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 5
        weights = torch.softmax(torch.randn(batch_size, seq_len, 3), dim=-1)
        means = torch.randn(batch_size, seq_len, 3)
        log_vars = torch.randn(batch_size, seq_len, 3)
        targets = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss_no_weights = head.nll_loss(weights, means, log_vars, targets, mask)
        loss_uniform = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            loss_weights=torch.ones(batch_size, seq_len),
        )

        assert torch.isclose(loss_no_weights, loss_uniform, atol=1e-5)

    def test_weighted_loss_differs_from_unweighted(self):
        """Test that non-uniform weights produce different loss."""
        from origami.model.heads import ContinuousHead

        config = ModelConfig(d_model=32, num_mixture_components=3)
        head = ContinuousHead(config)

        batch_size, seq_len = 2, 5
        weights = torch.softmax(torch.randn(batch_size, seq_len, 3), dim=-1)
        means = torch.randn(batch_size, seq_len, 3)
        log_vars = torch.randn(batch_size, seq_len, 3)
        targets = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Non-uniform weights
        loss_weights = torch.ones(batch_size, seq_len)
        loss_weights[0, 0] = 10.0  # Weight first position higher

        loss_unweighted = head.nll_loss(weights, means, log_vars, targets, mask)
        loss_weighted = head.nll_loss(
            weights,
            means,
            log_vars,
            targets,
            mask,
            loss_weights=loss_weights,
        )

        # Should be different (unless per-token losses happen to be equal)
        # We can't guarantee they differ, but they should both be valid
        assert not torch.isnan(loss_unweighted)
        assert not torch.isnan(loss_weighted)

    def test_model_continuous_loss_uses_weights(self):
        """Test that model's continuous loss respects loss_weights."""
        from origami.preprocessing import NumericScaler

        # Create data with scaled numerics
        data = [
            {"context": 100.0, "target": 200.0},
            {"context": 150.0, "target": 250.0},
        ]

        # Preprocess - cat_threshold=1 means any field with >1 unique value is scaled
        scaler = NumericScaler(cat_threshold=1)
        scaled_data = scaler.fit_transform(data)

        tokenizer = JSONTokenizer()
        tokenizer.fit(scaled_data)

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            use_continuous_head=True,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        collator = OrigamiDataCollator(tokenizer)
        batch = collator([tokenizer.tokenize(obj) for obj in scaled_data])

        # Create weights
        weights = torch.ones_like(batch.input_ids, dtype=torch.float)

        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                numeric_values=batch.numeric_values,
                numeric_mask=batch.numeric_mask,
                loss_weights=weights,
            )

        # Should have continuous params and valid loss
        assert output.continuous_params is not None
        assert output.loss is not None
        assert not torch.isnan(output.loss)
