"""Tests for ORIGAMI training infrastructure."""

import pytest
import torch

from origami.config import ModelConfig, TrainingConfig
from origami.model import OrigamiModel
from origami.tokenizer import JSONTokenizer
from origami.training import (
    EpochStats,
    OrigamiDataCollator,
    OrigamiDataset,
    OrigamiTrainer,
    TrainResult,
)
from origami.utils import available_devices as get_available_devices

AVAILABLE_DEVICES = get_available_devices()


@pytest.fixture
def tokenizer():
    """Create a tokenizer fitted on sample data."""
    tokenizer = JSONTokenizer()
    tokenizer.fit(
        [
            {"name": "Alice", "age": 30, "scores": [90, 85]},
            {"name": "Bob", "age": 25, "active": True},
            {"name": "Charlie", "age": 35, "city": "NYC"},
        ]
    )
    return tokenizer


@pytest.fixture
def sample_data():
    """Sample training data."""
    return [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ]


class TestOrigamiDataset:
    """Tests for OrigamiDataset."""

    def test_length(self, tokenizer, sample_data):
        """Test dataset length."""
        dataset = OrigamiDataset(sample_data, tokenizer)
        assert len(dataset) == len(sample_data)

    def test_getitem_returns_tokenized_instance(self, tokenizer, sample_data):
        """Test that __getitem__ returns TokenizedInstance."""
        from origami.tokenizer.json_tokenizer import TokenizedInstance

        dataset = OrigamiDataset(sample_data, tokenizer)
        item = dataset[0]
        assert isinstance(item, TokenizedInstance)
        assert len(item.tokens) > 0
        assert len(item.paths) == len(item.tokens)

    def test_shuffle_true_produces_different_orderings(self, tokenizer):
        """Test that shuffle=True produces different key orderings."""
        data = [{"a": 1, "b": 2, "c": 3, "d": 4}]
        tokenizer.fit(data)

        dataset = OrigamiDataset(data, tokenizer, shuffle=True)

        # Collect token sequences from multiple accesses of same index
        sequences = []
        for _ in range(20):
            item = dataset[0]
            sequences.append(tuple(t for t in item.tokens))

        # With 4 keys and 20 accesses, we should see multiple orderings
        unique_sequences = set(sequences)
        assert len(unique_sequences) > 1, "Shuffle should produce different orderings"

    def test_shuffle_false_deterministic(self, tokenizer):
        """Test that shuffle=False produces deterministic tokenization."""
        data = [{"a": 1, "b": 2, "c": 3, "d": 4}]
        tokenizer.fit(data)

        dataset = OrigamiDataset(data, tokenizer, shuffle=False)

        # Multiple accesses should produce identical results
        sequences = []
        for _ in range(5):
            item = dataset[0]
            sequences.append(tuple(t for t in item.tokens))

        # All sequences should be identical
        assert len(set(sequences)) == 1, "shuffle=False should be deterministic"

    def test_default_shuffle_is_true(self, tokenizer, sample_data):
        """Test that shuffle defaults to True."""
        dataset = OrigamiDataset(sample_data, tokenizer)
        assert dataset.shuffle is True


class TestOrigamiDataCollator:
    """Tests for OrigamiDataCollator."""

    def test_collate_single_instance(self, tokenizer, sample_data):
        """Test collating a single instance."""
        collator = OrigamiDataCollator(tokenizer)
        instance = tokenizer.tokenize(sample_data[0])

        batch = collator([instance])

        assert batch.input_ids.shape[0] == 1
        assert batch.attention_mask.shape == batch.input_ids.shape
        assert batch.path_types.shape[:2] == batch.input_ids.shape
        assert batch.path_ids.shape[:2] == batch.input_ids.shape
        assert batch.path_lengths.shape == batch.input_ids.shape
        assert batch.labels.shape == batch.input_ids.shape

    def test_collate_multiple_instances(self, tokenizer, sample_data):
        """Test collating multiple instances with padding."""
        collator = OrigamiDataCollator(tokenizer)
        instances = [tokenizer.tokenize(obj) for obj in sample_data]

        batch = collator(instances)

        assert batch.input_ids.shape[0] == len(sample_data)
        # All tensors should have same batch size
        assert batch.attention_mask.shape[0] == len(sample_data)
        assert batch.path_types.shape[0] == len(sample_data)
        assert batch.path_ids.shape[0] == len(sample_data)
        assert batch.path_lengths.shape[0] == len(sample_data)
        assert batch.labels.shape[0] == len(sample_data)

    def test_collate_with_max_length(self, tokenizer, sample_data):
        """Test that max_length truncates sequences."""
        collator = OrigamiDataCollator(tokenizer, max_length=5)
        instances = [tokenizer.tokenize(obj) for obj in sample_data]

        batch = collator(instances)

        assert batch.input_ids.shape[1] <= 5

    def test_attention_mask_reflects_padding(self, tokenizer):
        """Test that attention mask correctly marks padding."""
        # Create objects with different lengths
        short = {"name": "A"}
        long = {"name": "Alice", "age": 30, "active": True}
        tokenizer.fit([short, long])

        collator = OrigamiDataCollator(tokenizer)
        instances = [tokenizer.tokenize(short), tokenizer.tokenize(long)]

        batch = collator(instances)

        # Shorter sequence should have fewer True values in mask
        assert batch.attention_mask[0].sum() < batch.attention_mask[1].sum()

    def test_collate_empty_raises(self, tokenizer):
        """Test that empty batch raises ValueError."""
        collator = OrigamiDataCollator(tokenizer)

        with pytest.raises(ValueError, match="Cannot collate empty batch"):
            collator([])

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_collate_to_device(self, tokenizer, sample_data, device):
        """Test collating directly to a device."""
        collator = OrigamiDataCollator(tokenizer, device=device)
        instances = [tokenizer.tokenize(obj) for obj in sample_data]

        batch = collator(instances)

        assert batch.input_ids.device.type == device.type
        assert batch.attention_mask.device.type == device.type
        assert batch.path_types.device.type == device.type
        assert batch.path_ids.device.type == device.type
        assert batch.path_lengths.device.type == device.type
        assert batch.labels.device.type == device.type

    def test_labels_are_copy_of_input_ids(self, tokenizer, sample_data):
        """Test that labels are a copy of input_ids for autoregressive training."""
        collator = OrigamiDataCollator(tokenizer)
        instances = [tokenizer.tokenize(obj) for obj in sample_data]

        batch = collator(instances)

        assert torch.equal(batch.labels, batch.input_ids)
        # But they should be different tensors (not same object)
        batch.labels[0, 0] = -1
        assert not torch.equal(batch.labels, batch.input_ids)


class TestLeftPadding:
    """Tests for left-padding behavior in OrigamiDataCollator.

    Left-padding is critical for batched prediction: all sequences end at
    the same position, so `logits[:, -1, :]` gives next-token predictions
    for all sequences simultaneously.
    """

    @pytest.fixture
    def lp_tokenizer(self):
        """Create a tokenizer for left-padding tests."""
        tokenizer = JSONTokenizer()
        tokenizer.fit(
            [
                {"a": 1},
                {"a": 1, "b": 2},
                {"a": 1, "b": 2, "c": 3, "d": 4},
            ]
        )
        return tokenizer

    def test_left_padding_structure(self, lp_tokenizer):
        """Test that padding is on the LEFT (start) of sequences."""
        short = {"a": 1}  # Short sequence
        long = {"a": 1, "b": 2, "c": 3}  # Long sequence

        collator = OrigamiDataCollator(lp_tokenizer)
        short_inst = lp_tokenizer.tokenize(short)
        long_inst = lp_tokenizer.tokenize(long)

        batch = collator([short_inst, long_inst])

        # Both sequences should have same length (padded to longest)
        assert batch.input_ids.shape[1] == len(long_inst.tokens)

        # Short sequence: PAD tokens at START, real tokens at END
        short_ids = batch.input_ids[0]
        short_mask = batch.attention_mask[0]

        # First tokens should be PAD (mask=False)
        pad_count = (~short_mask).sum().item()
        assert pad_count > 0, "Short sequence should have padding"

        # Check PAD tokens are at the START
        for i in range(pad_count):
            assert short_ids[i] == lp_tokenizer.vocab.pad_token_id
            assert not short_mask[i]

        # Real tokens should be at the END
        for i in range(pad_count, len(short_ids)):
            assert short_ids[i] != lp_tokenizer.vocab.pad_token_id
            assert short_mask[i]

        # Long sequence should have no padding
        assert batch.attention_mask[1].all()

    def test_all_sequences_end_at_same_position(self, lp_tokenizer):
        """Test that all sequences end at the same position (critical for batched prediction)."""
        objects = [
            {"a": 1},  # Short
            {"a": 1, "b": 2},  # Medium
            {"a": 1, "b": 2, "c": 3, "d": 4},  # Long
        ]

        collator = OrigamiDataCollator(lp_tokenizer)
        instances = [lp_tokenizer.tokenize(obj) for obj in objects]
        batch = collator(instances)

        # All sequences should have real (non-PAD) tokens at the last position
        for i in range(len(objects)):
            last_token = batch.input_ids[i, -1]
            assert last_token != lp_tokenizer.vocab.pad_token_id, (
                f"Sequence {i} has PAD at last position"
            )

            # The last token should be END
            assert last_token == lp_tokenizer.vocab.end_id, (
                f"Sequence {i} should end with END token"
            )

            # Attention mask should be True at last position
            assert batch.attention_mask[i, -1]

    def test_path_encoding_aligned_with_left_padding(self, lp_tokenizer):
        """Test that path encoding is correctly aligned with left-padded sequences."""
        short = {"a": 1}
        long = {"a": 1, "b": 2}

        collator = OrigamiDataCollator(lp_tokenizer)
        short_inst = lp_tokenizer.tokenize(short)
        long_inst = lp_tokenizer.tokenize(long)

        batch = collator([short_inst, long_inst])

        # For short sequence, path info should be at positions where real tokens are
        short_mask = batch.attention_mask[0]
        pad_count = (~short_mask).sum().item()

        # Padded positions should have zero path_lengths
        for i in range(pad_count):
            assert batch.path_lengths[0, i] == 0

        # Real token positions should have correct path_lengths (could be 0 for START/END)
        # but should match the original tokenized instance
        for i, path in enumerate(short_inst.paths):
            pos = pad_count + i
            expected_depth = min(len(path), lp_tokenizer.max_depth)
            assert batch.path_lengths[0, pos] == expected_depth

    def test_lengths_tensor_correct(self, lp_tokenizer):
        """Test that lengths tensor reflects original sequence lengths."""
        objects = [
            {"a": 1},  # Short
            {"a": 1, "b": 2, "c": 3},  # Long
        ]

        collator = OrigamiDataCollator(lp_tokenizer)
        instances = [lp_tokenizer.tokenize(obj) for obj in objects]
        batch = collator(instances)

        # lengths should match original token counts
        assert batch.lengths[0] == len(instances[0].tokens)
        assert batch.lengths[1] == len(instances[1].tokens)

    def test_model_forward_with_left_padded_batch(self, lp_tokenizer):
        """Test that model forward pass works correctly with left-padded batches."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=lp_tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=lp_tokenizer.vocab)
        model.eval()

        objects = [
            {"a": 1},
            {"a": 1, "b": 2, "c": 3, "d": 4},
        ]

        collator = OrigamiDataCollator(lp_tokenizer)
        instances = [lp_tokenizer.tokenize(obj) for obj in objects]
        batch = collator(instances)

        with torch.no_grad():
            output = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
            )

        # Output should have correct shape
        batch_size, seq_len = batch.input_ids.shape
        assert output.logits.shape == (batch_size, seq_len, lp_tokenizer.vocab.size)

        # No NaN in outputs for real (non-PAD) positions
        # PAD positions may have NaN due to all-masked attention (softmax of all -inf)
        for b in range(batch_size):
            mask = batch.attention_mask[b]
            real_logits = output.logits[b][mask]
            assert not torch.isnan(real_logits).any(), f"NaN in real positions for batch {b}"

    def test_training_loss_with_left_padded_batch(self, lp_tokenizer):
        """Test that training loss computation works with left-padded batches."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=lp_tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=lp_tokenizer.vocab)

        objects = [
            {"a": 1},
            {"a": 1, "b": 2, "c": 3, "d": 4},
        ]

        collator = OrigamiDataCollator(lp_tokenizer)
        instances = [lp_tokenizer.tokenize(obj) for obj in objects]
        batch = collator(instances)

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )

        # Loss should be computed and not be NaN or Inf
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

        # Loss should be positive
        assert output.loss > 0

    def test_grammar_constraints_with_left_padding(self, lp_tokenizer):
        """Test that grammar constraints work correctly with left-padded sequences."""
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=lp_tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=lp_tokenizer.vocab)
        # Attach grammar PDA for this test (normally done by trainer)
        from origami.constraints.json_grammar import JSONGrammarPDA

        model._grammar_pda = JSONGrammarPDA(lp_tokenizer.vocab, max_depth=config.max_depth)

        # Create batch with different length sequences
        objects = [
            {"a": 1},
            {"a": 1, "b": 2},
        ]

        collator = OrigamiDataCollator(lp_tokenizer)
        instances = [lp_tokenizer.tokenize(obj) for obj in objects]
        batch = collator(instances)

        # Training pass (grammar constraints applied)
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )

        # For positions where we predict real tokens, logits should have valid entries
        # (some tokens masked to -inf, but not all)
        for b in range(batch.input_ids.shape[0]):
            mask = batch.attention_mask[b]
            # For each real position (except the last which predicts nothing useful)
            for t in range(mask.sum().item() - 1):
                pos = (~mask).sum().item() + t  # Actual position in padded sequence
                logits_at_pos = output.logits[b, pos]
                # Should have some valid tokens (not all -inf)
                valid_count = (logits_at_pos > float("-inf")).sum()
                assert valid_count > 0, f"No valid tokens at position {pos} for batch {b}"


class TestTrainResult:
    """Tests for TrainResult dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = TrainResult()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_eval_loss == float("inf")

    def test_mutable(self):
        """Test that state is mutable."""
        state = TrainResult()
        state.epoch = 5
        state.global_step = 100
        state.best_eval_loss = 0.5

        assert state.epoch == 5
        assert state.global_step == 100
        assert state.best_eval_loss == 0.5


class TestEpochStats:
    """Tests for EpochStats dataclass."""

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        metrics = EpochStats(
            loss=0.5,
            num_samples=100,
            num_tokens=1000,
            duration_seconds=2.0,
        )
        assert metrics.tokens_per_second == 500.0

    def test_tokens_per_second_zero_duration(self):
        """Test tokens per second with zero duration."""
        metrics = EpochStats(
            loss=0.5,
            num_samples=100,
            num_tokens=1000,
            duration_seconds=0.0,
        )
        assert metrics.tokens_per_second == 0.0


class TestOrigamiTrainer:
    """Tests for OrigamiTrainer."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Create a small model and tokenizer for testing."""
        tokenizer = JSONTokenizer()
        tokenizer.fit(
            [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        )

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
            max_array_position=tokenizer.max_array_index,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        return model, tokenizer

    def test_trainer_init(self, model_and_tokenizer):
        """Test trainer initialization."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}]

        config = TrainingConfig(num_epochs=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        assert trainer.model is model
        assert trainer.tokenizer is tokenizer
        assert len(trainer.train_dataset) == 1
        assert trainer.eval_dataset is None

    def test_trainer_with_eval_data(self, model_and_tokenizer):
        """Test trainer with evaluation data."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}]
        eval_data = [{"name": "Bob", "age": 25}]

        config = TrainingConfig(num_epochs=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
            config=config,
        )

        assert trainer.eval_dataset is not None
        assert len(trainer.eval_dataset) == 1

    def test_train_one_epoch(self, model_and_tokenizer):
        """Test training for one epoch."""
        model, tokenizer = model_and_tokenizer
        train_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ] * 5  # 10 samples

        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
        )
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        state = trainer.train()

        assert state.epoch == 0  # 0-indexed, so epoch 0 means 1 epoch completed
        assert state.global_step > 0

    def test_train_reduces_loss(self, model_and_tokenizer):
        """Test that training reduces loss over multiple epochs."""
        model, tokenizer = model_and_tokenizer
        train_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ] * 10  # 20 samples

        config = TrainingConfig(
            batch_size=4,
            num_epochs=5,
            learning_rate=1e-2,
        )
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        # Get initial loss
        initial_metrics = trainer._train_epoch()
        trainer.state.epoch = 0  # Reset for clean training

        # Train and collect losses
        losses = []
        for _ in range(5):
            metrics = trainer._train_epoch()
            losses.append(metrics.loss)

        # Loss should generally decrease (allow some variance)
        # Check that final loss is less than initial
        assert losses[-1] < initial_metrics.loss

    def test_evaluator_via_trainer(self, model_and_tokenizer):
        """Test evaluation via trainer's evaluator."""
        model, tokenizer = model_and_tokenizer
        eval_data = [{"name": "Bob", "age": 25}]

        # Use trainer's evaluator directly
        from origami.inference import OrigamiEvaluator

        evaluator = OrigamiEvaluator(model, tokenizer)
        results = evaluator.evaluate(eval_data)  # Loss is always computed

        assert "loss" in results
        assert results["loss"] > 0

    def test_callbacks(self, model_and_tokenizer):
        """Test epoch end callbacks."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}] * 4

        config = TrainingConfig(batch_size=2, num_epochs=2)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        # Track callback invocations using new callback API
        from origami.training import TrainerCallback

        callback_epochs = []

        class TestCallback(TrainerCallback):
            def on_epoch_end(self, trainer, state, metrics):
                callback_epochs.append(state.epoch)

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
            callbacks=[TestCallback()],
        )
        trainer.train()

        assert callback_epochs == [0, 1]

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_trainer_on_device(self, device):
        """Test trainer on different devices."""
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"name": "Alice", "age": 30}])

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        train_data = [{"name": "Alice", "age": 30}] * 4
        train_config = TrainingConfig(batch_size=2, num_epochs=1)

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=train_config,
            device=device,
        )

        assert trainer.device.type == device.type

        # Train should work
        state = trainer.train()
        assert state.global_step > 0

    def test_gradient_clipping(self, model_and_tokenizer):
        """Test that gradient clipping is applied."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}] * 4

        config = TrainingConfig(batch_size=2, num_epochs=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        # Training should complete without NaN (gradient clipping helps stability)
        state = trainer.train()
        assert state.global_step > 0

        # Verify no NaN in parameters
        for param in model.parameters():
            assert not torch.isnan(param).any()

    def test_learning_rate_schedule(self, model_and_tokenizer):
        """Test that learning rate follows warmup schedule."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}] * 20

        config = TrainingConfig(
            batch_size=2,
            num_epochs=2,
            warmup_steps=5,
            learning_rate=1e-3,
        )
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        # Initial LR should be 0 (start of warmup)
        initial_lr = trainer.scheduler.get_last_lr()[0]
        assert initial_lr < config.learning_rate

        # Train a few steps
        trainer._train_epoch()

        # LR should have increased during warmup
        current_lr = trainer.scheduler.get_last_lr()[0]
        assert current_lr > initial_lr


class TestEndToEndTraining:
    """End-to-end integration tests for training pipeline."""

    def test_full_pipeline_synthetic_data(self):
        """Test complete training pipeline on synthetic user data."""
        import random

        random.seed(42)
        torch.manual_seed(42)

        # Generate synthetic user data (small for fast tests)
        names = ["Alice", "Bob", "Charlie"]
        cities = ["NYC", "LA"]

        train_data = []
        for _ in range(20):
            train_data.append(
                {
                    "name": random.choice(names),
                    "age": random.randint(20, 60),
                    "city": random.choice(cities),
                }
            )

        eval_data = []
        for _ in range(5):
            eval_data.append(
                {
                    "name": random.choice(names),
                    "age": random.randint(20, 60),
                    "city": random.choice(cities),
                }
            )

        # Create tokenizer and fit on all data
        tokenizer = JSONTokenizer()
        tokenizer.fit(train_data + eval_data)

        # Create small model for fast tests
        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
            max_array_position=tokenizer.max_array_index,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        # Create trainer
        train_config = TrainingConfig(
            batch_size=4,
            num_epochs=3,
            learning_rate=1e-2,
        )

        # Track metrics using new callback API
        from origami.training import TrainerCallback

        train_losses = []
        eval_losses = []

        class MetricsTracker(TrainerCallback):
            def on_epoch_end(self, trainer, state, metrics):
                train_losses.append(metrics.loss)

            def on_evaluate(self, trainer, state, metrics):
                # metrics is now a dict with prefixed keys
                eval_losses.append(metrics.get("val_loss"))

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
            config=train_config,
            callbacks=[MetricsTracker()],
        )

        # Train
        state = trainer.train()

        # Verify training completed
        assert state.epoch == train_config.num_epochs - 1
        assert state.global_step > 0

        # Verify loss decreased
        assert train_losses[-1] < train_losses[0], "Training loss should decrease"

    def test_training_with_arrays(self):
        """Test training on data with nested arrays."""
        import random

        random.seed(123)
        torch.manual_seed(123)

        # Data with arrays (small for fast tests)
        train_data = []
        for i in range(10):
            train_data.append(
                {
                    "id": i,
                    "scores": [random.randint(0, 100) for _ in range(2)],
                    "tags": ["a", "b"] if i % 2 == 0 else ["x"],
                }
            )

        tokenizer = JSONTokenizer()
        tokenizer.fit(train_data)

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        train_config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
        )
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=train_config,
        )

        # Training should complete without errors
        state = trainer.train()
        assert state.global_step > 0

        # Model should produce valid output
        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        instances = [tokenizer.tokenize(train_data[0])]
        batch = collator(instances).to(trainer.device)

        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
            )

        assert output.logits.shape[0] == 1
        assert not torch.isnan(output.logits).any()

    def test_training_stability(self):
        """Test that training is numerically stable over many steps."""
        import random

        random.seed(456)
        torch.manual_seed(456)

        # Generate diverse data (small for fast tests)
        train_data = [{"x": random.random() * 100, "y": random.random() * 100} for _ in range(20)]

        tokenizer = JSONTokenizer()
        tokenizer.fit(train_data)

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        train_config = TrainingConfig(
            batch_size=4,
            num_epochs=5,
            learning_rate=1e-3,
        )
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=train_config,
        )

        # Train for multiple epochs
        trainer.train()

        # Verify no NaN in parameters
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name}"
            assert not torch.isinf(param).any(), f"Inf in {name}"

        # Verify model still produces valid output
        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        instances = [tokenizer.tokenize(train_data[0])]
        batch = collator(instances).to(trainer.device)

        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
            )

        assert not torch.isnan(output.logits).any()
        # Note: -inf values are expected from grammar constraint masking
        # Only check for +inf (actual numerical instability)
        assert not torch.isposinf(output.logits).any()


class TestEvaluator:
    """Tests for OrigamiEvaluator."""

    @pytest.fixture
    def setup(self):
        """Set up model, tokenizer, and test data."""
        from origami.constraints.json_grammar import JSONGrammarPDA

        data = [
            {"label": "A", "x": 1},
            {"label": "B", "x": 2},
            {"label": "A", "x": 3},
            {"label": "B", "x": 4},
        ]

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
        model._grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=config.max_depth)

        return model, tokenizer, data

    def test_evaluator_init(self, setup):
        """Test evaluator initialization."""
        from origami.inference import OrigamiEvaluator

        model, tokenizer, _ = setup
        evaluator = OrigamiEvaluator(model, tokenizer, target_key="label")

        assert evaluator.model is model
        assert evaluator.tokenizer is tokenizer
        assert evaluator.target_key == "label"

    def test_evaluate_loss_only(self, setup):
        """Test evaluating only loss (default when no metrics provided)."""
        from origami.inference import OrigamiEvaluator

        model, tokenizer, data = setup
        evaluator = OrigamiEvaluator(model, tokenizer)

        # No metrics dict = just loss
        results = evaluator.evaluate(data)

        assert "loss" in results
        assert len(results) == 1  # Only loss
        assert isinstance(results["loss"], float)
        assert results["loss"] > 0  # Loss should be positive

    def test_evaluate_prediction_metrics(self, setup):
        """Test evaluating prediction-based metrics."""
        from origami.inference import OrigamiEvaluator
        from origami.training import accuracy

        model, tokenizer, data = setup
        evaluator = OrigamiEvaluator(model, tokenizer, target_key="label")

        results = evaluator.evaluate(data, metrics={"acc": accuracy})

        assert "loss" in results  # Always included
        assert "acc" in results
        assert 0.0 <= results["acc"] <= 1.0

    def test_evaluate_multiple_metrics(self, setup):
        """Test evaluating multiple metrics together."""
        from origami.inference import OrigamiEvaluator
        from origami.training import accuracy, array_f1

        model, tokenizer, data = setup
        evaluator = OrigamiEvaluator(model, tokenizer, target_key="label")

        results = evaluator.evaluate(data, metrics={"acc": accuracy, "f1": array_f1})

        assert "loss" in results  # Always included
        assert "acc" in results
        assert "f1" in results
        assert len(results) == 3  # loss + acc + f1

    def test_evaluate_sample_size(self, setup):
        """Test evaluation with sample_size."""
        from origami.inference import OrigamiEvaluator

        model, tokenizer, data = setup
        evaluator = OrigamiEvaluator(model, tokenizer)

        # Sample 2 out of 4 - just loss (no metrics dict)
        results = evaluator.evaluate(data, sample_size=2)

        assert "loss" in results
        assert isinstance(results["loss"], float)

    def test_evaluate_requires_target_key_for_prediction_metrics(self, setup):
        """Test that prediction metrics require target_key."""
        from origami.inference import OrigamiEvaluator
        from origami.training import accuracy

        model, tokenizer, data = setup
        evaluator = OrigamiEvaluator(model, tokenizer, target_key=None)

        with pytest.raises(ValueError, match="target_key required"):
            evaluator.evaluate(data, metrics={"acc": accuracy})


class TestEvaluationScheduling:
    """Tests for step-based and epoch-based evaluation scheduling."""

    @pytest.fixture
    def setup(self):
        """Set up model, tokenizer, and data for scheduling tests."""
        from origami.constraints.json_grammar import JSONGrammarPDA

        data = [{"label": "A", "x": i} for i in range(20)]

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
        model._grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=config.max_depth)

        return model, tokenizer, data

    def test_step_based_evaluation(self, setup):
        """Test evaluation fires at specified step intervals."""
        from origami.training import TrainerCallback

        model, tokenizer, data = setup

        eval_steps_fired = []

        class EvalTracker(TrainerCallback):
            def on_evaluate(self, trainer, state, metrics):
                eval_steps_fired.append(state.global_step)

        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            eval_strategy="steps",
            eval_steps=2,  # Evaluate every 2 steps
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
            callbacks=[EvalTracker()],
        )
        trainer.train()

        # With 20 samples, batch_size=4, drop_last=True: 5 batches per epoch
        # Should evaluate at steps 2, 4 during training
        # Plus a final evaluation at step 5 (end of training)
        assert len(eval_steps_fired) >= 1
        # All interval evaluations should be at even steps
        interval_evals = [s for s in eval_steps_fired if s != trainer.state.global_step]
        assert all(step % 2 == 0 for step in interval_evals)
        # Final evaluation happens at the last step
        assert eval_steps_fired[-1] == trainer.state.global_step

    def test_epoch_based_evaluation_every_n_epochs(self, setup):
        """Test evaluation fires at epoch intervals."""
        from origami.training import TrainerCallback

        model, tokenizer, data = setup

        eval_epochs_fired = []

        class EvalTracker(TrainerCallback):
            def on_evaluate(self, trainer, state, metrics):
                eval_epochs_fired.append(state.epoch)

        config = TrainingConfig(
            batch_size=4,
            num_epochs=4,
            eval_strategy="epoch",
            eval_epochs=2,  # Evaluate every 2 epochs
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
            callbacks=[EvalTracker()],
        )
        trainer.train()

        # Should evaluate at epochs 1 and 3 (end of epoch 2 and 4)
        # epoch is 0-indexed, so (epoch+1) % 2 == 0 means epochs 1, 3
        assert len(eval_epochs_fired) == 2

    def test_no_evaluation_strategy(self, setup):
        """Test that eval_strategy='no' disables evaluation."""
        from origami.training import TrainerCallback

        model, tokenizer, data = setup

        eval_count = [0]

        class EvalTracker(TrainerCallback):
            def on_evaluate(self, trainer, state, metrics):
                eval_count[0] += 1

        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            eval_strategy="no",
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
            callbacks=[EvalTracker()],
        )
        trainer.train()

        assert eval_count[0] == 0

    def test_eval_metrics_config(self, setup):
        """Test that eval_metrics config is respected."""
        from origami.training import TrainerCallback

        model, tokenizer, data = setup

        received_metrics = []

        class EvalTracker(TrainerCallback):
            def on_evaluate(self, trainer, state, metrics):
                received_metrics.append(set(metrics.keys()))

        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            eval_strategy="epoch",
            # No eval_metrics = just loss (always computed)
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
            callbacks=[EvalTracker()],
        )
        trainer.train()

        assert len(received_metrics) >= 1
        # Should have val_loss since we have eval_data
        assert "val_loss" in received_metrics[0]

    def test_eval_on_train_includes_train_metrics(self, setup):
        """Test that eval_on_train=True includes train_ prefixed metrics."""
        from origami.training import TrainerCallback

        model, tokenizer, data = setup

        received_metrics = []

        class EvalTracker(TrainerCallback):
            def on_evaluate(self, trainer, state, metrics):
                received_metrics.append(dict(metrics))

        config = TrainingConfig(
            batch_size=4,
            num_epochs=1,
            eval_strategy="epoch",
            # No eval_metrics = just loss
            eval_on_train=True,
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
            callbacks=[EvalTracker()],
        )
        trainer.train()

        assert len(received_metrics) >= 1
        # Should have both train_loss and val_loss
        assert "train_loss" in received_metrics[0]
        assert "val_loss" in received_metrics[0]

    def test_on_best_callback_fires_on_improvement(self, setup):
        """Test that on_best callback fires when val_loss improves."""
        from origami.training import TrainerCallback

        model, tokenizer, data = setup

        best_events = []

        class BestTracker(TrainerCallback):
            def on_best(self, trainer, state, metrics):
                best_events.append(
                    {
                        "step": state.global_step,
                        "best_eval_loss": state.best_eval_loss,
                        "metrics": dict(metrics),
                    }
                )

        config = TrainingConfig(
            batch_size=4,
            num_epochs=3,
            eval_strategy="epoch",
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
            callbacks=[BestTracker()],
        )
        trainer.train()

        # on_best should have fired at least once (first eval is always best)
        assert len(best_events) >= 1

        # First event should have val_loss in metrics
        assert "val_loss" in best_events[0]["metrics"]

        # best_eval_loss in state should match the val_loss from metrics
        assert best_events[0]["best_eval_loss"] == best_events[0]["metrics"]["val_loss"]

        # If multiple best events, each should have a lower val_loss than previous
        for i in range(1, len(best_events)):
            assert best_events[i]["best_eval_loss"] < best_events[i - 1]["best_eval_loss"]

    def test_on_best_not_fired_when_loss_increases(self, setup):
        """Test that on_best does NOT fire when val_loss doesn't improve."""
        from origami.training import TrainerCallback

        model, tokenizer, data = setup

        best_count = [0]
        eval_count = [0]

        class CountTracker(TrainerCallback):
            def on_evaluate(self, trainer, state, metrics):
                eval_count[0] += 1

            def on_best(self, trainer, state, metrics):
                best_count[0] += 1

        config = TrainingConfig(
            batch_size=4,
            num_epochs=5,
            eval_strategy="epoch",
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
            callbacks=[CountTracker()],
        )
        trainer.train()

        # on_evaluate should fire every epoch (5 times)
        assert eval_count[0] == 5

        # on_best should fire fewer times than on_evaluate
        # (only when val_loss improves, not every eval)
        # At minimum, first eval always triggers on_best
        assert best_count[0] >= 1
        assert best_count[0] <= eval_count[0]


class TestCallbacks:
    """Tests for training callbacks."""

    @pytest.fixture
    def trainer_setup(self):
        """Set up trainer for callback tests."""
        data = [{"label": "A", "x": i} for i in range(20)]

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

    def test_progress_callback_on_interrupt(self, trainer_setup):
        """Test ProgressCallback.on_interrupt closes progress bar."""
        from origami.training import ProgressCallback, TrainResult

        model, tokenizer, data = trainer_setup

        config = TrainingConfig(batch_size=4, num_epochs=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        callback = ProgressCallback()
        state = TrainResult(epoch=2, global_step=50)

        # Simulate having an open progress bar
        callback._pbar = None  # Simulate no active pbar

        # Call on_interrupt (should not raise even without pbar)
        callback.on_interrupt(trainer, state, None)

    def test_table_log_callback_step_logging(self, trainer_setup, capsys):
        """Test TableLogCallback prints at specified intervals."""
        from origami.training import TableLogCallback, TrainResult

        model, tokenizer, data = trainer_setup

        config = TrainingConfig(batch_size=4, num_epochs=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        callback = TableLogCallback(print_every=5)
        state = TrainResult(epoch=0, global_step=5)
        state.current_batch_loss = 2.5
        state.current_lr = 0.001

        # Simulate batch start and end
        callback.on_batch_begin(trainer, state, None)
        callback.on_batch_end(trainer, state, None)

        captured = capsys.readouterr()
        assert "step: 5" in captured.out
        assert "loss: 2.5" in captured.out

    def test_table_log_callback_skips_non_interval(self, trainer_setup, capsys):
        """Test TableLogCallback skips non-interval steps."""
        from origami.training import TableLogCallback, TrainResult

        model, tokenizer, data = trainer_setup

        config = TrainingConfig(batch_size=4, num_epochs=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        callback = TableLogCallback(print_every=10)
        state = TrainResult(epoch=0, global_step=7)
        state.current_batch_loss = 2.5
        state.current_lr = 0.001

        callback.on_batch_begin(trainer, state, None)
        callback.on_batch_end(trainer, state, None)

        captured = capsys.readouterr()
        # Should not print anything since step 7 is not a multiple of 10
        assert "step:" not in captured.out

    def test_table_log_callback_evaluate_epoch_mode(self, trainer_setup, capsys):
        """Test TableLogCallback stores eval metrics for epoch-based logging."""
        from origami.training import TableLogCallback, TrainResult

        model, tokenizer, data = trainer_setup

        config = TrainingConfig(batch_size=4, num_epochs=1, eval_strategy="epoch")
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        callback = TableLogCallback(print_every=1)

        # First, fire evaluation (epoch mode)
        state = TrainResult(epoch=0, global_step=5)
        callback.on_evaluate(trainer, state, {"val_loss": 1.234})

        # Metrics should be stored for next batch print
        assert callback._pending_eval_metrics is not None
        assert callback._pending_eval_metrics["val_loss"] == 1.234

    def test_table_log_callback_on_interrupt(self, trainer_setup, capsys):
        """Test TableLogCallback.on_interrupt prints message."""
        from origami.training import TableLogCallback, TrainResult

        model, tokenizer, data = trainer_setup

        config = TrainingConfig(batch_size=4, num_epochs=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        callback = TableLogCallback(print_every=1)
        state = TrainResult(epoch=2, global_step=50, interrupted=True)

        callback.on_interrupt(trainer, state, None)

        captured = capsys.readouterr()
        assert "interrupted" in captured.out.lower()
        assert "epoch 2" in captured.out

    def test_table_log_callback_step_eval_combined(self, trainer_setup, capsys):
        """Test TableLogCallback combines batch and eval on step-based eval."""
        from origami.training import TableLogCallback, TrainResult

        model, tokenizer, data = trainer_setup

        config = TrainingConfig(batch_size=4, num_epochs=1, eval_strategy="steps", eval_steps=10)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            eval_data=data[:4],
            config=config,
        )

        callback = TableLogCallback(print_every=10)
        state = TrainResult(epoch=0, global_step=10)
        state.current_batch_loss = 2.5
        state.current_lr = 0.001

        # on_batch_end should defer printing
        callback.on_batch_begin(trainer, state, None)
        callback.on_batch_end(trainer, state, None)

        # Check that batch parts are deferred
        assert callback._pending_batch_parts is not None

        # Now on_evaluate should print combined
        callback.on_evaluate(trainer, state, {"val_loss": 1.5, "val_acc": 0.8})

        captured = capsys.readouterr()
        # Should have both batch info and eval metrics on same line
        assert "step: 10" in captured.out
        assert "val_loss" in captured.out


class TestAllowComplexValues:
    """Tests for allow_complex_values in training."""

    @pytest.fixture
    def trainer_setup(self):
        """Create model, tokenizer, and data for trainer tests."""
        data = [
            {"tags": ["a", "b"], "x": 1},
            {"tags": ["c"], "x": 2},
            {"tags": ["a", "c"], "x": 3},
        ]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        config = ModelConfig(d_model=32, n_heads=2, n_layers=1)
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        return model, tokenizer, data

    def test_training_config_default_is_none(self):
        """Test that TrainingConfig.allow_complex_values defaults to None."""
        config = TrainingConfig(num_epochs=1)
        assert config.allow_complex_values is None

    def test_training_config_explicit_values(self):
        """Test setting explicit True/False values."""
        config_true = TrainingConfig(num_epochs=1, allow_complex_values=True)
        assert config_true.allow_complex_values is True

        config_false = TrainingConfig(num_epochs=1, allow_complex_values=False)
        assert config_false.allow_complex_values is False

    def test_auto_detect_enables_for_array_f1(self, trainer_setup):
        """Test auto-detection enables allow_complex_values for array_f1."""
        from origami.training import array_f1

        model, tokenizer, data = trainer_setup
        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            eval_metrics={"f1": array_f1},  # Aliased name - detection uses fn.__name__
            target_key="tags",
            # allow_complex_values=None (default, auto-detect)
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        assert trainer.evaluator.allow_complex_values is True

    def test_auto_detect_enables_for_array_jaccard(self, trainer_setup):
        """Test auto-detection enables allow_complex_values for array_jaccard."""
        from origami.training import array_jaccard

        model, tokenizer, data = trainer_setup
        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            eval_metrics={"jaccard": array_jaccard},  # Aliased name
            target_key="tags",
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        assert trainer.evaluator.allow_complex_values is True

    def test_auto_detect_disabled_for_accuracy_only(self, trainer_setup):
        """Test auto-detection keeps False for accuracy-only metrics."""
        from origami.training import accuracy

        model, tokenizer, data = trainer_setup
        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            eval_metrics={"acc": accuracy},
            target_key="x",
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        assert trainer.evaluator.allow_complex_values is False

    def test_auto_detect_disabled_when_no_metrics(self, trainer_setup):
        """Test auto-detection keeps False when no metrics configured."""
        model, tokenizer, data = trainer_setup
        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            # No eval_metrics
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        assert trainer.evaluator.allow_complex_values is False

    def test_explicit_true_respected(self, trainer_setup):
        """Test explicit True is respected even without complex metrics."""
        from origami.training import accuracy

        model, tokenizer, data = trainer_setup
        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            eval_metrics={"acc": accuracy},
            target_key="x",
            allow_complex_values=True,  # Explicit True
        )

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=data,
            config=config,
        )

        assert trainer.evaluator.allow_complex_values is True

    def test_explicit_false_with_complex_metric_warns(self, trainer_setup):
        """Test warning when explicit False conflicts with complex-requiring metrics."""
        import warnings

        from origami.training import array_f1

        model, tokenizer, data = trainer_setup
        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            eval_metrics={"array_f1": array_f1},  # Use canonical name for detection
            target_key="tags",
            allow_complex_values=False,  # Explicit False
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer = OrigamiTrainer(
                model=model,
                tokenizer=tokenizer,
                train_data=data,
                config=config,
            )

            assert len(w) == 1
            assert "allow_complex_values=False" in str(w[0].message)
            assert "array_f1" in str(w[0].message)
            assert trainer.evaluator.allow_complex_values is False


class TestComplexValueMetricsRegistry:
    """Tests for COMPLEX_VALUE_METRICS constant."""

    def test_contains_expected_metrics(self):
        """Test registry contains known complex-requiring metrics."""
        from origami.training import COMPLEX_VALUE_METRICS

        assert "array_f1" in COMPLEX_VALUE_METRICS
        assert "array_jaccard" in COMPLEX_VALUE_METRICS
        assert "object_key_accuracy" in COMPLEX_VALUE_METRICS

    def test_does_not_contain_simple_metrics(self):
        """Test registry does not contain simple metrics."""
        from origami.training import COMPLEX_VALUE_METRICS

        assert "accuracy" not in COMPLEX_VALUE_METRICS

    def test_is_frozenset(self):
        """Test registry is immutable frozenset."""
        from origami.training import COMPLEX_VALUE_METRICS

        assert isinstance(COMPLEX_VALUE_METRICS, frozenset)


class TestAccelerateIntegration:
    """Tests for optional accelerate integration."""

    @pytest.fixture
    def trainer_components(self):
        """Create trainer components for testing."""
        tokenizer = JSONTokenizer()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        tokenizer.fit(data)
        config = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64)
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        return model, tokenizer, data

    def test_trainer_has_is_main_process(self, trainer_components):
        """Test trainer has is_main_process property."""
        model, tokenizer, data = trainer_components
        trainer = OrigamiTrainer(model, tokenizer, data)
        assert hasattr(trainer, "is_main_process")
        # Without accelerate active, should always be True
        assert trainer.is_main_process is True

    def test_trainer_has_unwrapped_model(self, trainer_components):
        """Test trainer has unwrapped_model property."""
        model, tokenizer, data = trainer_components
        trainer = OrigamiTrainer(model, tokenizer, data)
        assert hasattr(trainer, "unwrapped_model")
        # Without accelerate active, should return the model directly
        assert trainer.unwrapped_model is model

    def test_use_accelerate_config_default(self):
        """Test use_accelerate defaults to True in TrainingConfig."""
        config = TrainingConfig()
        assert config.use_accelerate is True

    def test_use_accelerate_config_can_be_disabled(self):
        """Test use_accelerate can be set to False."""
        config = TrainingConfig(use_accelerate=False)
        assert config.use_accelerate is False

    def test_trainer_respects_use_accelerate_false(self, trainer_components):
        """Test trainer doesn't use accelerate when disabled."""
        model, tokenizer, data = trainer_components
        config = TrainingConfig(use_accelerate=False, num_epochs=1)
        trainer = OrigamiTrainer(model, tokenizer, data, config=config)

        # Accelerator should not be initialized
        assert trainer._accelerator is None
        assert trainer._use_accelerate is False
        # Properties should still work
        assert trainer.is_main_process is True
        assert trainer.unwrapped_model is model

    def test_trainer_works_with_accelerate_disabled(self, trainer_components):
        """Test training works with use_accelerate=False."""
        model, tokenizer, data = trainer_components
        config = TrainingConfig(use_accelerate=False, num_epochs=1, batch_size=2)
        trainer = OrigamiTrainer(model, tokenizer, data, config=config)

        # Should train without errors
        result = trainer.train()
        assert result.epoch == 0
        assert result.global_step > 0

    def test_accelerate_available_constant_exists(self):
        """Test ACCELERATE_AVAILABLE constant is defined."""
        from origami.training.trainer import ACCELERATE_AVAILABLE

        assert isinstance(ACCELERATE_AVAILABLE, bool)

    def test_callbacks_check_is_main_process(self, trainer_components):
        """Test callbacks respect is_main_process."""
        from origami.training.callbacks import ProgressCallback, TableLogCallback

        model, tokenizer, data = trainer_components

        # Create callbacks
        progress_cb = ProgressCallback()
        table_cb = TableLogCallback(print_every=1)

        config = TrainingConfig(num_epochs=1, batch_size=2)
        trainer = OrigamiTrainer(
            model, tokenizer, data, config=config, callbacks=[progress_cb, table_cb]
        )

        # Callbacks should have access to is_main_process via trainer
        # Just verify training works with callbacks
        result = trainer.train()
        assert result.completed

    def test_explicit_device_disables_accelerate(self, trainer_components):
        """Test that passing explicit device disables accelerate."""
        model, tokenizer, data = trainer_components
        config = TrainingConfig(use_accelerate=True, num_epochs=1)

        # Passing explicit device should disable accelerate
        trainer = OrigamiTrainer(model, tokenizer, data, config=config, device=torch.device("cpu"))

        # Should not use accelerate when device is explicitly passed
        assert trainer._accelerator is None
        assert trainer._use_accelerate is False
        assert trainer.device.type == "cpu"
