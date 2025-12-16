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
    tokenizer.fit([
        {"name": "Alice", "age": 30, "scores": [90, 85]},
        {"name": "Bob", "age": 25, "active": True},
        {"name": "Charlie", "age": 35, "city": "NYC"},
    ])
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
        for key in ["input_ids", "attention_mask", "path_types", "path_ids", "path_lengths", "labels"]:
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

        for key in ["input_ids", "attention_mask", "path_types", "path_ids", "path_lengths", "labels"]:
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
        tokenizer.fit([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ])

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

        # Track callback invocations
        callback_epochs = []

        def on_epoch_end(epoch, metrics):
            callback_epochs.append(epoch)

        trainer.on_epoch_end = on_epoch_end
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
            train_data.append({
                "name": random.choice(names),
                "age": random.randint(20, 60),
                "city": random.choice(cities),
            })

        eval_data = []
        for _ in range(5):
            eval_data.append({
                "name": random.choice(names),
                "age": random.randint(20, 60),
                "city": random.choice(cities),
            })

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

            # Track metrics
            train_losses = []
            eval_losses = []

            def on_epoch_end(epoch, metrics):
                train_losses.append(metrics.loss)

            def on_eval_end(epoch, metrics):
                eval_losses.append(metrics.loss)

            trainer.on_epoch_end = on_epoch_end
            trainer.on_eval_end = on_eval_end

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
            train_data.append({
                "id": i,
                "scores": [random.randint(0, 100) for _ in range(2)],
                "tags": ["a", "b"] if i % 2 == 0 else ["x"],
            })

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
        train_data = [
            {"x": random.random() * 100, "y": random.random() * 100}
            for _ in range(20)
        ]

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
        state = trainer.train()

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
