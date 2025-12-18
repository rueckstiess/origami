"""Tests for ORIGAMI training infrastructure."""

import tempfile
from pathlib import Path

import pytest
import torch

from origami.model import OrigamiConfig, OrigamiModel, TrainingConfig
from origami.tokenizer import JSONTokenizer
from origami.training import (
    EvalDataset,
    OrigamiDataCollator,
    OrigamiTrainer,
    TrainMetrics,
    TrainState,
    UpscaledDataset,
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


class TestUpscaledDataset:
    """Tests for UpscaledDataset."""

    def test_length_with_upscale_factor(self, tokenizer, sample_data):
        """Test that length is multiplied by upscale factor."""
        dataset = UpscaledDataset(sample_data, tokenizer, upscale_factor=10)
        assert len(dataset) == len(sample_data) * 10

    def test_length_no_upscale(self, tokenizer, sample_data):
        """Test length with upscale_factor=1."""
        dataset = UpscaledDataset(sample_data, tokenizer, upscale_factor=1)
        assert len(dataset) == len(sample_data)

    def test_getitem_returns_tokenized_instance(self, tokenizer, sample_data):
        """Test that __getitem__ returns TokenizedInstance."""
        from origami.tokenizer.json_tokenizer import TokenizedInstance

        dataset = UpscaledDataset(sample_data, tokenizer, upscale_factor=1)
        item = dataset[0]
        assert isinstance(item, TokenizedInstance)
        assert len(item.tokens) > 0
        assert len(item.paths) == len(item.tokens)

    def test_upscale_maps_to_base_index(self, tokenizer, sample_data):
        """Test that upscaled indices map to correct base indices."""
        dataset = UpscaledDataset(sample_data, tokenizer, upscale_factor=5)

        # Indices 0-4 should all come from base item 0
        for i in range(5):
            base_item = dataset.get_base_item(i // 5)
            assert base_item == sample_data[0]

        # Indices 5-9 should come from base item 1
        for i in range(5, 10):
            base_item = dataset.get_base_item(i // 5)
            assert base_item == sample_data[1]

    def test_shuffle_produces_different_orderings(self, tokenizer):
        """Test that shuffle produces different key orderings."""
        data = [{"a": 1, "b": 2, "c": 3, "d": 4}]
        tokenizer.fit(data)

        dataset = UpscaledDataset(data, tokenizer, upscale_factor=20)

        # Collect token sequences from multiple accesses
        sequences = []
        for i in range(20):
            item = dataset[i]
            sequences.append(tuple(t for t in item.tokens))

        # With 4 keys and 20 samples, we should see multiple orderings
        unique_sequences = set(sequences)
        assert len(unique_sequences) > 1, "Shuffle should produce different orderings"

    def test_base_size_property(self, tokenizer, sample_data):
        """Test base_size property."""
        dataset = UpscaledDataset(sample_data, tokenizer, upscale_factor=10)
        assert dataset.base_size == len(sample_data)

    def test_invalid_upscale_factor_raises(self, tokenizer, sample_data):
        """Test that upscale_factor < 1 raises ValueError."""
        with pytest.raises(ValueError, match="upscale_factor must be >= 1"):
            UpscaledDataset(sample_data, tokenizer, upscale_factor=0)

        with pytest.raises(ValueError, match="upscale_factor must be >= 1"):
            UpscaledDataset(sample_data, tokenizer, upscale_factor=-1)


class TestEvalDataset:
    """Tests for EvalDataset."""

    def test_length(self, tokenizer, sample_data):
        """Test dataset length."""
        dataset = EvalDataset(sample_data, tokenizer)
        assert len(dataset) == len(sample_data)

    def test_getitem_returns_tokenized_instance(self, tokenizer, sample_data):
        """Test that __getitem__ returns TokenizedInstance."""
        from origami.tokenizer.json_tokenizer import TokenizedInstance

        dataset = EvalDataset(sample_data, tokenizer)
        item = dataset[0]
        assert isinstance(item, TokenizedInstance)

    def test_deterministic_tokenization(self, tokenizer):
        """Test that tokenization is deterministic (no shuffle)."""
        data = [{"a": 1, "b": 2, "c": 3, "d": 4}]
        tokenizer.fit(data)

        dataset = EvalDataset(data, tokenizer)

        # Multiple accesses should produce identical results
        sequences = []
        for _ in range(5):
            item = dataset[0]
            sequences.append(tuple(t for t in item.tokens))

        # All sequences should be identical
        assert len(set(sequences)) == 1, "Eval dataset should be deterministic"


class TestOrigamiDataCollator:
    """Tests for OrigamiDataCollator."""

    def test_collate_single_instance(self, tokenizer, sample_data):
        """Test collating a single instance."""
        collator = OrigamiDataCollator(tokenizer)
        instance = tokenizer.tokenize(sample_data[0])

        batch = collator([instance])

        assert batch["input_ids"].shape[0] == 1
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["path_types"].shape[:2] == batch["input_ids"].shape
        assert batch["path_ids"].shape[:2] == batch["input_ids"].shape
        assert batch["path_lengths"].shape == batch["input_ids"].shape
        assert batch["labels"].shape == batch["input_ids"].shape

    def test_collate_multiple_instances(self, tokenizer, sample_data):
        """Test collating multiple instances with padding."""
        collator = OrigamiDataCollator(tokenizer)
        instances = [tokenizer.tokenize(obj) for obj in sample_data]

        batch = collator(instances)

        assert batch["input_ids"].shape[0] == len(sample_data)
        # All tensors should have same batch size
        for key in [
            "input_ids",
            "attention_mask",
            "path_types",
            "path_ids",
            "path_lengths",
            "labels",
        ]:
            assert batch[key].shape[0] == len(sample_data)

    def test_collate_with_max_length(self, tokenizer, sample_data):
        """Test that max_length truncates sequences."""
        collator = OrigamiDataCollator(tokenizer, max_length=5)
        instances = [tokenizer.tokenize(obj) for obj in sample_data]

        batch = collator(instances)

        assert batch["input_ids"].shape[1] <= 5

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
        assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()

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

        for key in [
            "input_ids",
            "attention_mask",
            "path_types",
            "path_ids",
            "path_lengths",
            "labels",
        ]:
            assert batch[key].device.type == device.type

    def test_labels_are_copy_of_input_ids(self, tokenizer, sample_data):
        """Test that labels are a copy of input_ids for autoregressive training."""
        collator = OrigamiDataCollator(tokenizer)
        instances = [tokenizer.tokenize(obj) for obj in sample_data]

        batch = collator(instances)

        assert torch.equal(batch["labels"], batch["input_ids"])
        # But they should be different tensors (not same object)
        batch["labels"][0, 0] = -1
        assert not torch.equal(batch["labels"], batch["input_ids"])


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
        assert batch["input_ids"].shape[1] == len(long_inst.tokens)

        # Short sequence: PAD tokens at START, real tokens at END
        short_ids = batch["input_ids"][0]
        short_mask = batch["attention_mask"][0]

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
        assert batch["attention_mask"][1].all()

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
            last_token = batch["input_ids"][i, -1]
            assert last_token != lp_tokenizer.vocab.pad_token_id, (
                f"Sequence {i} has PAD at last position"
            )

            # The last token should be END
            assert last_token == lp_tokenizer.vocab.end_id, (
                f"Sequence {i} should end with END token"
            )

            # Attention mask should be True at last position
            assert batch["attention_mask"][i, -1]

    def test_path_encoding_aligned_with_left_padding(self, lp_tokenizer):
        """Test that path encoding is correctly aligned with left-padded sequences."""
        short = {"a": 1}
        long = {"a": 1, "b": 2}

        collator = OrigamiDataCollator(lp_tokenizer)
        short_inst = lp_tokenizer.tokenize(short)
        long_inst = lp_tokenizer.tokenize(long)

        batch = collator([short_inst, long_inst])

        # For short sequence, path info should be at positions where real tokens are
        short_mask = batch["attention_mask"][0]
        pad_count = (~short_mask).sum().item()

        # Padded positions should have zero path_lengths
        for i in range(pad_count):
            assert batch["path_lengths"][0, i] == 0

        # Real token positions should have correct path_lengths (could be 0 for START/END)
        # but should match the original tokenized instance
        for i, path in enumerate(short_inst.paths):
            pos = pad_count + i
            expected_depth = min(len(path), lp_tokenizer.max_depth)
            assert batch["path_lengths"][0, pos] == expected_depth

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
        assert batch["lengths"][0] == len(instances[0].tokens)
        assert batch["lengths"][1] == len(instances[1].tokens)

    def test_model_forward_with_left_padded_batch(self, lp_tokenizer):
        """Test that model forward pass works correctly with left-padded batches."""
        config = OrigamiConfig(
            vocab_size=lp_tokenizer.vocab.size,
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
                input_ids=batch["input_ids"],
                path_types=batch["path_types"],
                path_ids=batch["path_ids"],
                path_lengths=batch["path_lengths"],
                attention_mask=batch["attention_mask"],
            )

        # Output should have correct shape
        batch_size, seq_len = batch["input_ids"].shape
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

        # No NaN in outputs for real (non-PAD) positions
        # PAD positions may have NaN due to all-masked attention (softmax of all -inf)
        for b in range(batch_size):
            mask = batch["attention_mask"][b]
            real_logits = output.logits[b][mask]
            assert not torch.isnan(real_logits).any(), f"NaN in real positions for batch {b}"

    def test_training_loss_with_left_padded_batch(self, lp_tokenizer):
        """Test that training loss computation works with left-padded batches."""
        config = OrigamiConfig(
            vocab_size=lp_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=lp_tokenizer.max_depth,
            use_grammar_constraints=True,
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
            input_ids=batch["input_ids"],
            path_types=batch["path_types"],
            path_ids=batch["path_ids"],
            path_lengths=batch["path_lengths"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Loss should be computed and not be NaN or Inf
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

        # Loss should be positive
        assert output.loss > 0

    def test_grammar_constraints_with_left_padding(self, lp_tokenizer):
        """Test that grammar constraints work correctly with left-padded sequences."""
        config = OrigamiConfig(
            vocab_size=lp_tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=lp_tokenizer.max_depth,
            use_grammar_constraints=True,
        )
        model = OrigamiModel(config, vocab=lp_tokenizer.vocab)

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
            input_ids=batch["input_ids"],
            path_types=batch["path_types"],
            path_ids=batch["path_ids"],
            path_lengths=batch["path_lengths"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # For positions where we predict real tokens, logits should have valid entries
        # (some tokens masked to -inf, but not all)
        for b in range(batch["input_ids"].shape[0]):
            mask = batch["attention_mask"][b]
            # For each real position (except the last which predicts nothing useful)
            for t in range(mask.sum().item() - 1):
                pos = (~mask).sum().item() + t  # Actual position in padded sequence
                logits_at_pos = output.logits[b, pos]
                # Should have some valid tokens (not all -inf)
                valid_count = (logits_at_pos > float("-inf")).sum()
                assert valid_count > 0, f"No valid tokens at position {pos} for batch {b}"


class TestTrainState:
    """Tests for TrainState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = TrainState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_eval_loss == float("inf")

    def test_mutable(self):
        """Test that state is mutable."""
        state = TrainState()
        state.epoch = 5
        state.global_step = 100
        state.best_eval_loss = 0.5

        assert state.epoch == 5
        assert state.global_step == 100
        assert state.best_eval_loss == 0.5


class TestTrainMetrics:
    """Tests for TrainMetrics dataclass."""

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        metrics = TrainMetrics(
            loss=0.5,
            num_samples=100,
            num_tokens=1000,
            duration_seconds=2.0,
        )
        assert metrics.tokens_per_second == 500.0

    def test_tokens_per_second_zero_duration(self):
        """Test tokens per second with zero duration."""
        metrics = TrainMetrics(
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

        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
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

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
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

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
        )

        assert trainer.eval_dataset is not None
        assert len(trainer.eval_dataset) == 1

    def test_trainer_with_upscale(self, model_and_tokenizer):
        """Test trainer with upscaling."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}]

        config = TrainingConfig(upscale_factor=10)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=config,
        )

        assert len(trainer.train_dataset) == 10

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

    def test_evaluate(self, model_and_tokenizer):
        """Test evaluation."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}]
        eval_data = [{"name": "Bob", "age": 25}]

        config = TrainingConfig(batch_size=1)
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
            config=config,
        )

        metrics = trainer.evaluate()

        assert isinstance(metrics, TrainMetrics)
        assert metrics.loss > 0
        assert metrics.num_samples == 1
        assert metrics.num_tokens > 0

    def test_evaluate_without_eval_data_raises(self, model_and_tokenizer):
        """Test that evaluate raises without eval data."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}]

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
        )

        with pytest.raises(ValueError, match="No eval dataset provided"):
            trainer.evaluate()

    def test_save_and_load_checkpoint(self, model_and_tokenizer):
        """Test checkpoint save and load."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}] * 4

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(batch_size=2, num_epochs=2)
            trainer = OrigamiTrainer(
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                config=config,
                checkpoint_dir=tmpdir,
            )

            # Train a bit
            trainer._train_epoch()
            trainer.state.epoch = 1
            trainer.state.best_eval_loss = 0.5

            # Save checkpoint
            path = trainer.save_checkpoint("test")
            assert Path(path).exists()

            # Modify state
            trainer.state.epoch = 999
            trainer.state.best_eval_loss = 999.0

            # Load checkpoint
            trainer.load_checkpoint(path)

            assert trainer.state.epoch == 1
            assert trainer.state.best_eval_loss == 0.5

    def test_save_checkpoint_without_dir_raises(self, model_and_tokenizer):
        """Test that save_checkpoint raises without checkpoint_dir."""
        model, tokenizer = model_and_tokenizer
        train_data = [{"name": "Alice", "age": 30}]

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
        )

        with pytest.raises(ValueError, match="No checkpoint directory"):
            trainer.save_checkpoint("test")

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

        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
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
        import tempfile

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
        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_depth=tokenizer.max_depth,
            max_array_position=tokenizer.max_array_index,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer
            train_config = TrainingConfig(
                batch_size=4,
                num_epochs=3,
                learning_rate=1e-2,
                upscale_factor=2,  # 2x upscaling for augmentation
                save_every_n_epochs=3,
            )
            trainer = OrigamiTrainer(
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                eval_data=eval_data,
                config=train_config,
                checkpoint_dir=tmpdir,
            )

            # Track metrics using new callback API
            from origami.training import TrainerCallback

            train_losses = []
            eval_losses = []

            class MetricsTracker(TrainerCallback):
                def on_epoch_end(self, trainer, state, metrics):
                    train_losses.append(metrics.loss)

                def on_evaluate(self, trainer, state, metrics):
                    eval_losses.append(metrics.loss)

            trainer = OrigamiTrainer(
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                eval_data=eval_data,
                config=train_config,
                checkpoint_dir=tmpdir,
                callbacks=[MetricsTracker()],
            )

            # Train
            state = trainer.train()

            # Verify training completed
            assert state.epoch == train_config.num_epochs - 1
            assert state.global_step > 0

            # Verify loss decreased
            assert train_losses[-1] < train_losses[0], "Training loss should decrease"

            # Verify checkpoint was saved
            checkpoint_path = Path(tmpdir) / "epoch_3.pt"
            assert checkpoint_path.exists()

            # Verify best model was saved
            best_path = Path(tmpdir) / "best.pt"
            assert best_path.exists()

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

        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
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
        batch = tokenizer.encode_batch([train_data[0]])
        batch = batch.to(trainer.device)

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

        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
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
        batch = tokenizer.encode_batch([train_data[0]])
        batch = batch.to(trainer.device)

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
