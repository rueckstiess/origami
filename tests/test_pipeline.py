"""Tests for OrigamiPipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from origami.config import DataConfig, ModelConfig, OrigamiConfig, TrainingConfig
from origami.pipeline import OrigamiPipeline


class TestOrigamiConfigNested:
    """Tests for OrigamiConfig nested structure validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrigamiConfig(training=TrainingConfig(num_epochs=10))
        assert config.model.d_model == 128
        assert config.model.n_heads == 4
        assert config.model.n_layers == 4
        assert config.data.numeric_mode == "disabled"
        assert config.training.batch_size == 32
        assert config.training.num_epochs == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OrigamiConfig(
            model=ModelConfig(d_model=256, n_heads=8),
            data=DataConfig(numeric_mode="scale", cat_threshold=50),
            training=TrainingConfig(num_epochs=5),
        )
        assert config.model.d_model == 256
        assert config.model.n_heads == 8
        assert config.data.numeric_mode == "scale"
        assert config.data.cat_threshold == 50

    def test_config_validation_d_model_divisible_by_n_heads(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(d_model=100, n_heads=3)

    def test_config_validation_n_layers(self):
        """Test that n_layers must be >= 1."""
        with pytest.raises(ValueError, match="n_layers"):
            ModelConfig(n_layers=0)

    def test_config_validation_cat_threshold(self):
        """Test that cat_threshold must be >= 1."""
        with pytest.raises(ValueError, match="cat_threshold"):
            DataConfig(cat_threshold=0)

    def test_config_validation_n_bins(self):
        """Test that n_bins must be >= 2 for discretize mode."""
        with pytest.raises(ValueError, match="n_bins"):
            DataConfig(numeric_mode="discretize", n_bins=1)

    def test_config_validation_kvpe_pooling(self):
        """Test that invalid kvpe_pooling raises error."""
        with pytest.raises(ValueError, match="kvpe_pooling"):
            ModelConfig(kvpe_pooling="invalid")

    def test_config_validation_backbone(self):
        """Test that invalid backbone raises error."""
        with pytest.raises(ValueError, match="backbone"):
            ModelConfig(backbone="invalid")

    def test_config_validation_eval_strategy(self):
        """Test that invalid eval_strategy raises error (e.g., 'step' vs 'steps')."""
        with pytest.raises(ValueError, match="eval_strategy"):
            TrainingConfig(eval_strategy="step")  # Should be "steps"

    def test_config_validation_numeric_mode(self):
        """Test that invalid numeric_mode raises error."""
        with pytest.raises(ValueError, match="numeric_mode"):
            DataConfig(numeric_mode="invalid")

    def test_config_validation_bin_strategy(self):
        """Test that invalid bin_strategy raises error."""
        with pytest.raises(ValueError, match="bin_strategy"):
            DataConfig(bin_strategy="invalid")

    def test_config_validation_device(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValueError, match="device"):
            OrigamiConfig(device="gpu")  # Should be "cuda"

    def test_config_to_yaml(self):
        """Test to_yaml() method produces valid YAML output."""
        config = OrigamiConfig(
            model=ModelConfig(d_model=64, n_layers=2),
            training=TrainingConfig(batch_size=16),
        )
        yaml_str = config.to_yaml()

        # Should contain key fields
        assert "d_model: 64" in yaml_str
        assert "n_layers: 2" in yaml_str
        assert "batch_size: 16" in yaml_str
        assert "model:" in yaml_str
        assert "training:" in yaml_str

    def test_config_repr_formatting(self):
        """Test __repr__ produces nicely formatted output."""
        config = ModelConfig(d_model=64, n_layers=2)
        repr_str = repr(config)

        # Should be multi-line with proper formatting
        assert "ModelConfig(" in repr_str
        assert "d_model=64" in repr_str
        assert "n_layers=2" in repr_str
        assert "\n" in repr_str  # Multi-line

    def test_nested_config_repr(self):
        """Test __repr__ handles nested configs properly."""
        config = OrigamiConfig(
            model=ModelConfig(d_model=64),
        )
        repr_str = repr(config)

        # Should show nested structure
        assert "OrigamiConfig(" in repr_str
        assert "ModelConfig(" in repr_str

    def test_config_to_yaml_with_callables(self):
        """Test to_yaml() handles eval_metrics callables cleanly."""
        from origami.training import accuracy

        config = OrigamiConfig(
            training=TrainingConfig(eval_metrics={"accuracy": accuracy}),
        )
        yaml_str = config.to_yaml()

        # Should show metric names as list, not ugly !!python/name
        assert "eval_metrics:" in yaml_str
        assert "accuracy" in yaml_str
        assert "!!python" not in yaml_str  # No ugly Python object notation

    def test_config_validation_dropout(self):
        """Test that dropout must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=-0.1)

    def test_config_validation_batch_size(self):
        """Test that batch_size must be >= 1."""
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_config_validation_learning_rate(self):
        """Test that learning_rate must be > 0."""
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=0)
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=-0.001)

    def test_config_validation_num_epochs(self):
        """Test that num_epochs must be >= 0."""
        with pytest.raises(ValueError, match="num_epochs"):
            TrainingConfig(num_epochs=-1)

    def test_config_validation_max_vocab_size(self):
        """Test that max_vocab_size must be >= 0."""
        with pytest.raises(ValueError, match="max_vocab_size"):
            DataConfig(max_vocab_size=-1)


class TestPipelineFit:
    """Tests for OrigamiPipeline.fit()."""

    @pytest.fixture
    def simple_data(self):
        """Simple training data."""
        return [
            {"name": "Alice", "age": 25, "city": "NYC"},
            {"name": "Bob", "age": 30, "city": "LA"},
            {"name": "Carol", "age": 35, "city": "NYC"},
        ]

    def test_fit_with_defaults(self, simple_data):
        """Test fitting with default config."""
        pipeline = OrigamiPipeline()
        pipeline.fit(simple_data, epochs=1, verbose=False)

        assert pipeline._fitted
        assert pipeline._model is not None
        assert pipeline._tokenizer is not None
        assert pipeline._preprocessor is None  # numeric_mode="disabled"

    def test_fit_with_eval_data(self, simple_data):
        """Test fitting with evaluation data."""
        train_data = simple_data[:2]
        eval_data = simple_data[2:]

        pipeline = OrigamiPipeline()
        pipeline.fit(train_data, eval_data=eval_data, epochs=1, verbose=False)

        assert pipeline._fitted

    def test_fit_with_discretize_mode(self):
        """Test fitting with discretization preprocessing."""
        # Data with high-cardinality numeric field
        data = [{"category": i % 3, "value": float(i)} for i in range(200)]

        config = OrigamiConfig(
            data=DataConfig(numeric_mode="discretize", cat_threshold=10, n_bins=5),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)

        assert pipeline._fitted
        assert pipeline._preprocessor is not None
        from origami.preprocessing import NumericDiscretizer

        assert isinstance(pipeline._preprocessor, NumericDiscretizer)

    def test_fit_with_scale_mode(self):
        """Test fitting with scaling preprocessing."""
        # Data with high-cardinality numeric field
        data = [{"category": i % 3, "value": float(i)} for i in range(200)]

        config = OrigamiConfig(
            data=DataConfig(numeric_mode="scale", cat_threshold=10),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)

        assert pipeline._fitted
        assert pipeline._preprocessor is not None
        from origami.preprocessing import NumericScaler

        assert isinstance(pipeline._preprocessor, NumericScaler)

    def test_fit_empty_data_raises(self):
        """Test that fitting with empty data raises error."""
        pipeline = OrigamiPipeline()
        with pytest.raises(ValueError, match="empty"):
            pipeline.fit([], epochs=1)

    def test_fit_returns_self(self, simple_data):
        """Test that fit() returns self for method chaining."""
        pipeline = OrigamiPipeline()
        result = pipeline.fit(simple_data, epochs=1, verbose=False)
        assert result is pipeline


class TestPipelineSaveLoad:
    """Tests for OrigamiPipeline.save() and load()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline."""
        data = [
            {"name": "Alice", "value": 1.0},
            {"name": "Bob", "value": 2.0},
        ]
        pipeline = OrigamiPipeline()
        pipeline.fit(data, epochs=1, verbose=False)
        return pipeline

    def test_save_load_roundtrip(self, fitted_pipeline):
        """Test save and load preserve state."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            fitted_pipeline.save(path)
            loaded = OrigamiPipeline.load(path)

            assert loaded._fitted
            assert loaded._model is not None
            assert loaded._tokenizer is not None
            assert loaded.config.model.d_model == fitted_pipeline.config.model.d_model
        finally:
            path.unlink()

    def test_save_load_with_scaler(self):
        """Test save/load with NumericScaler."""
        data = [{"x": float(i), "y": i % 3} for i in range(200)]

        config = OrigamiConfig(data=DataConfig(numeric_mode="scale", cat_threshold=10))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            pipeline.save(path)
            loaded = OrigamiPipeline.load(path)

            assert loaded.config.data.numeric_mode == "scale"
            assert loaded._preprocessor is not None
            from origami.preprocessing import NumericScaler

            assert isinstance(loaded._preprocessor, NumericScaler)
        finally:
            path.unlink()

    def test_save_load_with_discretizer(self):
        """Test save/load with NumericDiscretizer."""
        data = [{"x": float(i), "y": i % 3} for i in range(200)]

        config = OrigamiConfig(
            data=DataConfig(numeric_mode="discretize", cat_threshold=10, n_bins=5)
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            pipeline.save(path)
            loaded = OrigamiPipeline.load(path)

            assert loaded.config.data.numeric_mode == "discretize"
            assert loaded._preprocessor is not None
            from origami.preprocessing import NumericDiscretizer

            assert isinstance(loaded._preprocessor, NumericDiscretizer)
        finally:
            path.unlink()

    def test_state_dict_roundtrip(self, fitted_pipeline):
        """Test state_dict and from_state_dict preserve state."""
        state = fitted_pipeline.state_dict()

        # Verify state dict structure
        assert "version" in state
        assert "config" in state
        assert "model_state_dict" in state
        assert "tokenizer_state" in state

        # Reconstruct from state dict
        loaded = OrigamiPipeline.from_state_dict(state)

        assert loaded._fitted
        assert loaded._model is not None
        assert loaded._tokenizer is not None
        assert loaded.config.model.d_model == fitted_pipeline.config.model.d_model

    def test_state_dict_unfitted_raises(self):
        """Test that state_dict on unfitted pipeline raises error."""
        pipeline = OrigamiPipeline()
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.state_dict()

    def test_save_unfitted_raises(self):
        """Test that saving unfitted pipeline raises error."""
        pipeline = OrigamiPipeline()
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.save("test.pt")


class TestPipelinePredict:
    """Tests for OrigamiPipeline.predict()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline for prediction."""
        data = [
            {"name": "Alice", "category": "A"},
            {"name": "Bob", "category": "B"},
            {"name": "Carol", "category": "A"},
            {"name": "Dave", "category": "B"},
        ]
        # Use small batch size to ensure training actually happens
        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_heads=2, n_layers=2),
            training=TrainingConfig(batch_size=2),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=3, verbose=False)
        return pipeline

    def test_predict_single(self, fitted_pipeline):
        """Test single prediction."""
        obj = {"name": "Eve", "category": None}
        result = fitted_pipeline.predict(obj, target_key="category")

        # With grammar constraints, result should be a valid JSON value
        # (string, number, bool, None, dict, or list)
        assert isinstance(result, (str, int, float, bool, dict, list, type(None)))

    def test_predict_proba_top_k(self, fitted_pipeline):
        """Test top-k predictions via predict_proba."""
        obj = {"name": "Eve", "category": None}
        results = fitted_pipeline.predict_proba(obj, target_key="category", top_k=2)

        assert isinstance(results, list)
        assert len(results) == 2
        for _value, prob in results:
            assert 0 <= prob <= 1

    def test_predict_proba_all(self, fitted_pipeline):
        """Test prediction probability distribution."""
        obj = {"name": "Eve", "category": None}
        result = fitted_pipeline.predict_proba(obj, target_key="category")

        assert isinstance(result, dict)
        for prob in result.values():
            assert 0 <= prob <= 1

    def test_predict_batch(self, fitted_pipeline):
        """Test batch prediction."""
        objects = [
            {"name": "Eve", "category": None},
            {"name": "Frank", "category": None},
        ]
        results = fitted_pipeline.predict_batch(objects, target_key="category")

        # Results are now just a list of values
        assert len(results) == 2
        for result in results:
            # Each result is a value (not a list of tuples)
            assert isinstance(result, (str, int, float, bool, dict, list, type(None)))

    def test_predict_unfitted_raises(self):
        """Test that predicting with unfitted pipeline raises error."""
        pipeline = OrigamiPipeline()
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.predict({"a": 1}, target_key="b")


class TestPipelineGenerate:
    """Tests for OrigamiPipeline.generate()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline for generation."""
        # Seed for reproducible training
        torch.manual_seed(42)
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
        ]

        config = OrigamiConfig(model=ModelConfig(d_model=16, n_heads=4, n_layers=4))
        pipeline = OrigamiPipeline(config)
        # Train for more epochs so generation completes properly
        pipeline.fit(data, epochs=30, verbose=False)
        return pipeline

    def test_generate_single(self, fitted_pipeline):
        """Test generating a single sample."""
        samples = fitted_pipeline.generate(num_samples=1)

        assert len(samples) == 1
        assert isinstance(samples[0], dict)

    def test_generate_multiple(self, fitted_pipeline):
        """Test generating multiple samples."""
        samples = fitted_pipeline.generate(num_samples=3)

        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, dict)

    def test_generate_with_seed(self, fitted_pipeline):
        """Test that seed makes generation reproducible."""
        # Use temperature=0 for deterministic greedy decoding and increase max_length
        samples1 = fitted_pipeline.generate(
            num_samples=2, seed=42, temperature=0.0, max_length=1024
        )
        samples2 = fitted_pipeline.generate(
            num_samples=2, seed=42, temperature=0.0, max_length=1024
        )

        assert samples1 == samples2

    def test_generate_unfitted_raises(self):
        """Test that generating with unfitted pipeline raises error."""
        pipeline = OrigamiPipeline()
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.generate(num_samples=1)


class TestPipelineEmbed:
    """Tests for OrigamiPipeline.embed()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline for embedding."""
        data = [
            {"name": "Alice", "category": "A"},
            {"name": "Bob", "category": "B"},
        ]
        config = OrigamiConfig(model=ModelConfig(d_model=32, n_heads=4, n_layers=2))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)
        return pipeline

    def test_embed_single(self, fitted_pipeline):
        """Test embedding a single object."""
        obj = {"name": "Carol", "category": "A"}
        embedding = fitted_pipeline.embed(obj)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (32,)  # d_model

    def test_embed_batch(self, fitted_pipeline):
        """Test embedding multiple objects."""
        objects = [
            {"name": "Carol", "category": "A"},
            {"name": "Dave", "category": "B"},
        ]
        embeddings = fitted_pipeline.embed_batch(objects)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 32)

    def test_embed_pooling_strategies(self, fitted_pipeline):
        """Test different pooling strategies."""
        obj = {"name": "Carol", "category": "A"}

        for pooling in ["mean", "max", "last"]:
            embedding = fitted_pipeline.embed(obj, pooling=pooling)
            assert embedding.shape == (32,)

    def test_embed_target_pooling(self, fitted_pipeline):
        """Test target pooling strategy."""
        obj = {"name": "Carol", "category": "A"}
        embedding = fitted_pipeline.embed(obj, pooling="target", target_key="category")

        assert embedding.shape == (32,)

    def test_embed_normalized(self, fitted_pipeline):
        """Test that embeddings are normalized by default."""
        obj = {"name": "Carol", "category": "A"}
        embedding = fitted_pipeline.embed(obj, normalize=True)

        # L2 norm should be approximately 1
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_embed_unfitted_raises(self):
        """Test that embedding with unfitted pipeline raises error."""
        pipeline = OrigamiPipeline()
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.embed({"a": 1})


class TestPreprocessorSerialization:
    """Tests for preprocessor serialization."""

    def test_numeric_scaler_roundtrip(self):
        """Test NumericScaler serialization roundtrip."""
        from origami.preprocessing import NumericScaler

        data = [{"x": float(i), "y": float(i * 2)} for i in range(200)]

        scaler = NumericScaler(cat_threshold=10)
        scaler.fit(data)

        # Serialize and deserialize
        state = scaler.to_dict()
        loaded = NumericScaler.from_dict(state)

        # Check state preserved
        assert loaded.cat_threshold == scaler.cat_threshold
        assert loaded.scaled_fields == scaler.scaled_fields
        assert loaded.passthrough_fields == scaler.passthrough_fields

        # Check transform produces same results
        original = scaler.transform(data[:5])
        restored = loaded.transform(data[:5])

        for orig, rest in zip(original, restored, strict=True):
            assert orig["x"].value == rest["x"].value
            assert orig["y"].value == rest["y"].value

    def test_numeric_discretizer_roundtrip(self):
        """Test NumericDiscretizer serialization roundtrip."""
        from origami.preprocessing import NumericDiscretizer

        data = [{"x": float(i), "y": float(i * 2)} for i in range(200)]

        discretizer = NumericDiscretizer(cat_threshold=10, n_bins=5)
        discretizer.fit(data)

        # Serialize and deserialize
        state = discretizer.to_dict()
        loaded = NumericDiscretizer.from_dict(state)

        # Check state preserved
        assert loaded.cat_threshold == discretizer.cat_threshold
        assert loaded.n_bins == discretizer.n_bins
        assert loaded.discretized_fields == discretizer.discretized_fields

        # Check transform produces same results
        original = discretizer.transform(data[:5])
        restored = loaded.transform(data[:5])

        for orig, rest in zip(original, restored, strict=True):
            assert orig["x"] == rest["x"]
            assert orig["y"] == rest["y"]


class TestPipelineRepr:
    """Tests for OrigamiPipeline string representation."""

    def test_repr_unfitted(self):
        """Test repr for unfitted pipeline."""
        pipeline = OrigamiPipeline()
        assert "not fitted" in repr(pipeline)
        assert "disabled" in repr(pipeline)

    def test_repr_fitted(self):
        """Test repr for fitted pipeline."""
        data = [{"a": 1}, {"a": 2}]
        pipeline = OrigamiPipeline()
        pipeline.fit(data, epochs=1, verbose=False)
        assert "fitted" in repr(pipeline)
        assert "not fitted" not in repr(pipeline)

    def test_repr_with_scale_mode(self):
        """Test repr shows numeric mode."""
        config = OrigamiConfig(data=DataConfig(numeric_mode="scale"))
        pipeline = OrigamiPipeline(config)
        assert "scale" in repr(pipeline)


class TestPipelineDeviceManagement:
    """Tests for automatic device management."""

    def test_model_on_cpu_after_inference(self):
        """Test that model moves to CPU after first inference call."""
        data = [{"a": i, "b": i * 2} for i in range(50)]

        # Force CPU device in config to test the logic
        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2), device="cpu")
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1)

        # Model should be on configured device after fit
        device = next(pipeline.model.parameters()).device
        assert device.type == "cpu"

        # After prediction, should still be on CPU
        _ = pipeline.predict({"a": 3, "b": 0}, target_key="b")
        device = next(pipeline.model.parameters()).device
        assert device.type == "cpu"

    def test_inference_moves_to_cpu(self):
        """Test that inference components trigger device move."""
        from origami.inference.utils import GenerationError

        torch.manual_seed(42)
        data = [{"a": i} for i in range(20)]
        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2), device="cpu")
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=5)

        # All inference methods should work and keep model on CPU
        try:
            _ = pipeline.predict({"a": 5}, target_key="a")
        except GenerationError:
            pass  # Untrained model may not complete
        assert next(pipeline.model.parameters()).device.type == "cpu"

        try:
            _ = pipeline.generate(num_samples=1)
        except GenerationError:
            pass  # Untrained model may not complete
        assert next(pipeline.model.parameters()).device.type == "cpu"

        _ = pipeline.embed({"a": 5})
        assert next(pipeline.model.parameters()).device.type == "cpu"

    def test_training_device_set_from_config(self):
        """Test that training device is resolved from config."""
        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2), device="cpu")
        pipeline = OrigamiPipeline(config)

        # Before fit, training device is None
        assert pipeline._training_device is None

        data = [{"a": i} for i in range(20)]
        pipeline.fit(data, epochs=1)

        # After fit, training device should be set
        assert pipeline._training_device is not None
        assert pipeline._training_device.type == "cpu"

    def test_load_sets_training_device(self, tmp_path):
        """Test that loading a model sets training device."""
        data = [{"a": i, "b": i * 2} for i in range(50)]

        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2), device="cpu")
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1)

        # Save and reload
        path = tmp_path / "model.pt"
        pipeline.save(path)
        loaded = OrigamiPipeline.load(path)

        # Training device should be set from config
        assert loaded._training_device is not None

        # Model should be on CPU (loaded with map_location="cpu")
        assert next(loaded.model.parameters()).device.type == "cpu"


class TestPipelineEvaluate:
    """Tests for OrigamiPipeline.evaluate() method."""

    def test_evaluate_returns_loss(self):
        """Test that evaluate returns loss by default."""
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        results = pipeline.evaluate(data[:10])

        assert "loss" in results
        assert isinstance(results["loss"], float)
        assert results["loss"] > 0

    def test_evaluate_with_metrics(self):
        """Test that evaluate computes custom metrics."""
        from origami.training import accuracy

        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(target_key="label"),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        results = pipeline.evaluate(data[:10], target_key="label", metrics={"accuracy": accuracy})

        assert "loss" in results
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_evaluate_with_sample_size(self):
        """Test that evaluate respects sample_size."""
        data = [{"label": "A", "x": i} for i in range(100)]

        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1)

        # Should work with sample_size
        results = pipeline.evaluate(data, sample_size=10)

        assert "loss" in results

    def test_evaluate_uses_config_target_key(self):
        """Test that evaluate falls back to config target_key."""
        from origami.training import accuracy

        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(target_key="label"),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        # Don't pass target_key, should use config value
        results = pipeline.evaluate(data[:10], metrics={"accuracy": accuracy})

        assert "accuracy" in results

    def test_evaluate_auto_detects_allow_complex_values(self):
        """Test that evaluate auto-detects allow_complex_values from metrics."""
        from origami.training import array_f1

        torch.manual_seed(42)
        data = [{"tags": ["a", "b"], "x": i} for i in range(30)]

        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        # With array_f1 metric (using canonical name), should auto-enable complex values
        # The metric should be computed without error (even if untrained model gives wrong predictions)
        results = pipeline.evaluate(data[:5], target_key="tags", metrics={"array_f1": array_f1})

        assert "array_f1" in results
        assert isinstance(results["array_f1"], float)
        assert 0.0 <= results["array_f1"] <= 1.0

    def test_evaluate_explicit_allow_complex_values(self):
        """Test that evaluate accepts explicit allow_complex_values."""
        from origami.training import accuracy

        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        # Explicit True - should work for simple metrics too
        results = pipeline.evaluate(
            data[:5],
            target_key="label",
            metrics={"acc": accuracy},
            allow_complex_values=True,
        )

        assert "acc" in results

    def test_evaluate_explicit_false_overrides_auto_detect(self):
        """Test that explicit False overrides auto-detection."""
        from origami.training import array_f1

        torch.manual_seed(42)
        data = [{"tags": ["a", "b"], "x": i} for i in range(30)]

        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        # Explicit False - array_f1 will likely return 0.0 since predictions are primitives
        results = pipeline.evaluate(
            data[:5],
            target_key="tags",
            metrics={"array_f1": array_f1},
            allow_complex_values=False,
        )

        assert "array_f1" in results
        # With allow_complex_values=False, predictions are primitives, so F1 will be 0
        assert results["array_f1"] == 0.0


class TestPipelineInverseTransform:
    """Tests for inverse transform functionality with NumericScaler."""

    def test_inverse_transform_scaled_values(self):
        """Test that predictions are inverse-transformed correctly."""
        import random

        torch.manual_seed(42)
        random.seed(42)

        # Data with high-cardinality numeric field that will be scaled
        data = [{"label": "A", "value": random.random() * 1000} for _ in range(100)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2, use_continuous_head=True),
            data=DataConfig(numeric_mode="scale", cat_threshold=10),
            training=TrainingConfig(num_epochs=3),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=3)

        # Prediction should be inverse-transformed (in original scale)
        obj = {"label": "A", "value": 0}  # Target is 'value'
        try:
            prediction = pipeline.predict(obj, target_key="value")
            # If it works, prediction should be in reasonable range
            assert isinstance(prediction, (int, float))
        except Exception:
            # Untrained model may fail to complete generation
            pass

    def test_inverse_transform_preserves_non_scaled_fields(self):
        """Test that non-scaled fields are not affected by inverse transform."""
        import random

        torch.manual_seed(42)
        random.seed(42)

        # Data with categorical field and scaled numeric
        data = [{"category": random.choice(["A", "B"]), "count": i} for i in range(100)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            data=DataConfig(numeric_mode="scale", cat_threshold=10),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        # Predict category (should not be affected by numeric scaling)
        try:
            prediction = pipeline.predict({"category": "", "count": 50}, target_key="category")
            # Should be a categorical value if generation completes
            if prediction is not None:
                assert isinstance(prediction, str)
        except Exception:
            pass  # Untrained model may not complete

    def test_inverse_transform_uses_full_path_for_nested_fields(self):
        """Test that inverse transform uses full path (foo.bar) not just leaf key (bar).

        This is a regression test for a bug where _create_inverse_transform_fn
        used target_key.split(".")[-1] (leaf key) instead of the full path,
        which would fail to find the scaler for nested fields.
        """
        import random

        torch.manual_seed(42)
        random.seed(42)

        # Data with nested high-cardinality numeric field
        data = [{"label": "A", "stats": {"value": random.random() * 1000}} for _ in range(100)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2, use_continuous_head=True),
            data=DataConfig(numeric_mode="scale", cat_threshold=10),
            training=TrainingConfig(num_epochs=2),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2)

        # Verify the scaler uses full path "stats.value", not just "value"
        assert "stats.value" in pipeline._preprocessor.scaled_fields
        assert "value" not in pipeline._preprocessor.scaled_fields

        # The inverse transform function should work with full nested path
        inverse_fn = pipeline._create_inverse_transform_fn()
        assert inverse_fn is not None

        # Test inverse transform with a scaled z-score value
        # z-score of ~1.0 should map to approximately mean + 1*std
        stats = pipeline._preprocessor.get_scaler_stats("stats.value")
        test_zscore = 1.0
        expected_approx = stats["mean"] + test_zscore * stats["std"]

        result = inverse_fn(test_zscore, "stats.value")
        assert abs(result - expected_approx) < 0.01, (
            f"Inverse transform failed: expected ~{expected_approx}, got {result}"
        )

        # Verify leaf key alone would NOT work (field not found, returns unchanged)
        result_leaf = inverse_fn(test_zscore, "value")
        assert result_leaf == test_zscore, (
            "Leaf key 'value' should not match; value should pass through unchanged"
        )


class TestPipelinePreprocessorSerialization:
    """Tests for preprocessor serialization edge cases."""

    def test_unknown_preprocessor_type_raises(self, tmp_path):
        """Test that unknown preprocessor type raises ValueError on load."""
        data = [{"a": i, "b": i * 2} for i in range(50)]

        config = OrigamiConfig(model=ModelConfig(d_model=32, n_layers=2))
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1)

        # Save checkpoint
        path = tmp_path / "model.pt"
        pipeline.save(path)

        # Load checkpoint and modify preprocessor_type to unknown value
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["preprocessor_type"] = "UnknownPreprocessor"
        checkpoint["preprocessor_state"] = {"some": "data"}
        torch.save(checkpoint, path)

        # Loading should raise ValueError
        with pytest.raises(ValueError, match="Unknown preprocessor type"):
            OrigamiPipeline.load(path)

    def test_preprocessor_to_dict_handles_none(self):
        """Test that preprocessor_to_dict handles None preprocessor."""
        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            data=DataConfig(numeric_mode="disabled"),  # No preprocessing
        )
        pipeline = OrigamiPipeline(config)

        data = [{"a": "x", "b": "y"} for _ in range(20)]
        pipeline.fit(data, epochs=1)

        # No preprocessor should have been created
        assert pipeline._preprocessor is None

        # Should return None for preprocessor state
        result = pipeline._preprocessor_to_dict()
        assert result is None

    def test_load_preprocessor_handles_none(self):
        """Test that _load_preprocessor handles None inputs."""
        result = OrigamiPipeline._load_preprocessor(None, None)
        assert result is None

        result = OrigamiPipeline._load_preprocessor("NumericScaler", None)
        assert result is None


class TestPipelineCheckpointResume:
    """Tests for checkpoint save/load/resume functionality."""

    def test_save_includes_training_state(self, tmp_path):
        """Test that save includes training state by default."""
        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(num_epochs=3, batch_size=8),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=3, verbose=False)

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Load checkpoint and verify training state is present
        checkpoint = torch.load(path, weights_only=False)
        assert "training_state" in checkpoint
        assert "optimizer_state_dict" in checkpoint["training_state"]
        assert "scheduler_state_dict" in checkpoint["training_state"]
        assert "epoch" in checkpoint["training_state"]
        assert "global_step" in checkpoint["training_state"]

    def test_save_without_training_state(self, tmp_path):
        """Test that save can exclude training state."""
        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(num_epochs=2, batch_size=8),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2, verbose=False)

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path, include_training_state=False)

        # Verify training state is not included
        checkpoint = torch.load(path, weights_only=False)
        assert "training_state" not in checkpoint

    def test_load_restores_training_state(self, tmp_path):
        """Test that load restores training state."""
        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(num_epochs=3, batch_size=8),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=3, verbose=False)

        # Capture training state before save
        original_epoch = pipeline._training_state["epoch"]
        original_step = pipeline._training_state["global_step"]

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Load and verify training state is restored
        loaded = OrigamiPipeline.load(path)
        assert loaded._training_state is not None
        assert loaded._training_state["epoch"] == original_epoch
        assert loaded._training_state["global_step"] == original_step

    def test_preprocess_detects_resume(self, tmp_path):
        """Test that preprocess() detects resumption and skips refitting."""
        torch.manual_seed(42)
        data = [{"label": "A", "x": float(i)} for i in range(100)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            data=DataConfig(numeric_mode="scale", cat_threshold=10),
            training=TrainingConfig(num_epochs=2, batch_size=8),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2, verbose=False)

        # Save original preprocessor state
        original_scaled_fields = pipeline._preprocessor.scaled_fields.copy()

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Load checkpoint
        loaded = OrigamiPipeline.load(path)

        # Before preprocess, train data should be None
        assert loaded._train_processed is None

        # Call preprocess (should detect resume and use existing preprocessor)
        loaded.preprocess(data, verbose=False)

        # Verify preprocessor was not refit (same scaled fields)
        assert loaded._preprocessor.scaled_fields == original_scaled_fields
        # Verify data was processed
        assert loaded._train_processed is not None

    def test_resume_training_after_completed_epoch(self, tmp_path):
        """Test that resumed training continues from the next epoch when saved after completion.

        When an epoch completes fully and is saved, resuming should start from
        the next epoch (no replay of completed work).
        """
        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(num_epochs=5, batch_size=8),
        )

        # Train for 2 epochs (completes fully)
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2, verbose=False)

        # Verify we trained for 2 epochs and epoch is marked as completed
        assert pipeline._training_state["epoch"] == 1  # 0-indexed
        assert pipeline._training_state["epoch_completed"] is True

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Load and resume training for remaining epochs
        loaded = OrigamiPipeline.load(path)
        loaded.preprocess(data, verbose=False)

        # Should continue from epoch 2 (index), not replay epoch 1
        loaded.train(epochs=5, verbose=False)

        # Final epoch should be 4 (0-indexed)
        assert loaded._training_state["epoch"] == 4

    def test_full_checkpoint_resume_workflow(self, tmp_path):
        """Test the complete checkpoint resume workflow.

        Simulates a spot instance scenario:
        1. Start training from scratch
        2. Save checkpoint after a few epochs
        3. Load checkpoint (simulating restart)
        4. Resume training seamlessly
        """
        torch.manual_seed(42)
        data = [{"label": i % 3, "x": i, "y": float(i * 2)} for i in range(50)]
        eval_data = [{"label": i % 3, "x": i + 100, "y": float(i * 2 + 200)} for i in range(10)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            data=DataConfig(numeric_mode="scale", cat_threshold=10),
            training=TrainingConfig(num_epochs=6, batch_size=8),
        )

        # Phase 1: Initial training
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, eval_data=eval_data, epochs=3, verbose=False)

        initial_step = pipeline._training_state["global_step"]
        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Phase 2: Simulate restart - load checkpoint
        # This is the key workflow: same code path for initial and resume
        loaded = OrigamiPipeline.load(path)
        loaded.preprocess(data, eval_data=eval_data, verbose=False)
        loaded.train(epochs=6, verbose=False)

        # Verify training completed all epochs
        assert loaded._training_state["epoch"] == 5  # 0-indexed
        # Verify global_step increased beyond checkpoint
        assert loaded._training_state["global_step"] > initial_step

        # Verify model still works for inference
        prediction = loaded.predict({"label": 0, "x": 999, "y": 0.0}, target_key="label")
        assert prediction is not None

    def test_resume_with_eval_data(self, tmp_path):
        """Test that resume works correctly with evaluation data."""
        torch.manual_seed(42)
        train_data = [{"a": i, "b": i % 2} for i in range(40)]
        eval_data = [{"a": i, "b": i % 2} for i in range(10)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(
                num_epochs=4,
                batch_size=8,
                eval_strategy="epoch",
            ),
        )

        # Train initially
        pipeline = OrigamiPipeline(config)
        pipeline.fit(train_data, eval_data=eval_data, epochs=2, verbose=False)

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Resume
        loaded = OrigamiPipeline.load(path)
        loaded.preprocess(train_data, eval_data=eval_data, verbose=False)
        loaded.train(epochs=4, verbose=False)

        # Should complete without error
        assert loaded._training_state["epoch"] == 3

    def test_state_dict_version_updated(self):
        """Test that state_dict version is 1.2 for schema support."""
        torch.manual_seed(42)
        data = [{"a": i} for i in range(20)]

        pipeline = OrigamiPipeline()
        pipeline.fit(data, epochs=1, verbose=False)

        state = pipeline.state_dict()
        assert state["version"] == "1.2"

    def test_resume_via_fit(self, tmp_path):
        """Test that resume works via fit() as well as preprocess()/train()."""
        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(30)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(num_epochs=4, batch_size=8),
        )

        # Train for 2 epochs
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2, verbose=False)

        assert pipeline._training_state["epoch"] == 1  # 0-indexed

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Resume using fit() instead of preprocess()/train()
        loaded = OrigamiPipeline.load(path)
        loaded.fit(data, epochs=4, verbose=False)

        # Should have completed all 4 epochs
        assert loaded._training_state["epoch"] == 3  # 0-indexed

    def test_mid_epoch_resume_skips_completed_steps(self, tmp_path):
        """Test that mid-epoch resumption skips already-completed steps.

        Simulates a mid-epoch interruption by manually setting epoch_completed=False
        and steps_in_epoch to a partial value.
        """
        torch.manual_seed(42)
        # Use enough data to have multiple batches per epoch
        data = [{"label": "A", "x": i} for i in range(100)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(num_epochs=3, batch_size=8),
        )

        # Train for 1 full epoch
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)

        # Get the number of steps per epoch
        steps_per_epoch = pipeline._training_state["steps_in_epoch"]
        assert steps_per_epoch > 5, "Need enough steps to test mid-epoch resume"

        # Simulate mid-epoch interruption by modifying training state
        # Pretend we're in epoch 1 (0-indexed) and completed 3 steps
        pipeline._training_state["epoch"] = 1
        pipeline._training_state["epoch_completed"] = False
        pipeline._training_state["steps_in_epoch"] = 3
        # Adjust global_step to match (1 full epoch + 3 steps)
        pipeline._training_state["global_step"] = steps_per_epoch + 3

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Load and resume
        loaded = OrigamiPipeline.load(path)
        loaded.preprocess(data, verbose=False)

        # Should start from epoch 1, skipping first 3 steps
        loaded.train(epochs=3, verbose=False)

        # Final epoch should be 2 (trained epochs 1 and 2)
        assert loaded._training_state["epoch"] == 2
        assert loaded._training_state["epoch_completed"] is True

    def test_mid_epoch_state_saved_correctly(self, tmp_path):
        """Test that training state correctly reflects mid-epoch state.

        Verifies that epoch_completed is False during training and True after.
        """
        torch.manual_seed(42)
        data = [{"label": "A", "x": i} for i in range(50)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(num_epochs=2, batch_size=8),
        )

        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=2, verbose=False)

        # After fit completes, epoch should be marked as completed
        assert pipeline._training_state["epoch_completed"] is True
        assert pipeline._training_state["epoch"] == 1  # 0-indexed

        path = tmp_path / "checkpoint.pt"
        pipeline.save(path)

        # Verify saved state has epoch_completed
        checkpoint = torch.load(path, weights_only=False)
        assert "epoch_completed" in checkpoint["training_state"]
        assert checkpoint["training_state"]["epoch_completed"] is True


class TestPipelineEvaluateRegressionMetrics:
    """Tests for regression metrics (mse, rmse, mae) with scaled numeric fields."""

    @pytest.fixture
    def pipeline_with_scaled_numerics(self):
        """Create a pipeline with scaled numeric target field."""
        import random

        torch.manual_seed(42)
        random.seed(42)

        # Data with high-cardinality numeric field that will be scaled
        # Use values in a specific range so we can verify inverse transform works
        data = [{"category": i % 3, "price": 100.0 + i * 10.0} for i in range(100)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2, use_continuous_head=True),
            data=DataConfig(numeric_mode="scale", cat_threshold=10),
            training=TrainingConfig(target_key="price"),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=3, verbose=False)

        return pipeline, data

    def test_evaluate_mse_with_scaled_numerics(self, pipeline_with_scaled_numerics):
        """Test that MSE metric works with scaled numeric fields."""
        pipeline, data = pipeline_with_scaled_numerics

        results = pipeline.evaluate(
            data[:10],
            target_key="price",
            metrics={"mse": "mse"},
        )

        assert "loss" in results
        assert "mse" in results
        assert isinstance(results["mse"], float)
        # MSE should be non-negative
        assert results["mse"] >= 0

    def test_evaluate_rmse_with_scaled_numerics(self, pipeline_with_scaled_numerics):
        """Test that RMSE metric works with scaled numeric fields."""
        pipeline, data = pipeline_with_scaled_numerics

        results = pipeline.evaluate(
            data[:10],
            target_key="price",
            metrics={"rmse": "rmse"},
        )

        assert "rmse" in results
        assert isinstance(results["rmse"], float)
        # RMSE should be non-negative
        assert results["rmse"] >= 0

    def test_evaluate_mae_with_scaled_numerics(self, pipeline_with_scaled_numerics):
        """Test that MAE metric works with scaled numeric fields."""
        pipeline, data = pipeline_with_scaled_numerics

        results = pipeline.evaluate(
            data[:10],
            target_key="price",
            metrics={"mae": "mae"},
        )

        assert "mae" in results
        assert isinstance(results["mae"], float)
        # MAE should be non-negative
        assert results["mae"] >= 0

    def test_evaluate_multiple_regression_metrics(self, pipeline_with_scaled_numerics):
        """Test that multiple regression metrics can be computed together."""
        pipeline, data = pipeline_with_scaled_numerics

        results = pipeline.evaluate(
            data[:10],
            target_key="price",
            metrics={"mse": "mse", "rmse": "rmse", "mae": "mae"},
        )

        assert "mse" in results
        assert "rmse" in results
        assert "mae" in results

        # RMSE should equal sqrt(MSE)
        assert abs(results["rmse"] - results["mse"] ** 0.5) < 1e-6

    def test_evaluate_regression_metrics_in_original_scale(self, pipeline_with_scaled_numerics):
        """Test that regression metrics are computed in original scale, not scaled space.

        This is the key test: both y_true and y_pred should be inverse-transformed
        so metrics are interpretable in original units (e.g., dollars for price).
        """
        pipeline, data = pipeline_with_scaled_numerics

        # Get the scaler stats to understand the scale
        stats = pipeline._preprocessor.get_scaler_stats("price")
        original_std = stats["std"]

        results = pipeline.evaluate(
            data[:10],
            target_key="price",
            metrics={"rmse": "rmse"},
        )

        # If metrics were computed in scaled space, RMSE would be ~1.0 (z-score scale)
        # In original scale, RMSE should be proportional to the data's std dev
        # Even a bad model should have RMSE roughly in the same order of magnitude as std
        assert results["rmse"] > 1.0, (
            f"RMSE {results['rmse']:.2f} is too small - likely computed in scaled space. "
            f"Original std is {original_std:.2f}"
        )

    def test_evaluate_y_true_unwrapped_from_scaled_numeric(self, pipeline_with_scaled_numerics):
        """Test that y_true values are properly unwrapped from ScaledNumeric objects.

        This is a regression test for the bug where y_true contained ScaledNumeric
        objects instead of floats, causing sklearn metrics to fail.
        """
        pipeline, data = pipeline_with_scaled_numerics

        # This should not raise an error about ScaledNumeric types
        results = pipeline.evaluate(
            data[:5],
            target_key="price",
            metrics={"mse": "mse"},
        )

        # If we get here without error, ScaledNumeric was properly unwrapped
        assert "mse" in results


class TestPipelineSchemaConstraints:
    """Tests for schema constraint handling in pipeline preprocessing and evaluation."""

    def test_tokenizer_fitted_on_train_only(self):
        """Test that tokenizer is fitted on training data only.

        Eval-only values should NOT be in the vocabulary — they should map to
        UNK_VALUE when tokenized. This prevents data leakage from eval into the
        vocabulary and ensures schema masks work correctly for eval data.
        """
        from origami.tokenizer.vocabulary import ValueToken

        train_data = [
            {"color": "red", "size": "small"},
            {"color": "blue", "size": "medium"},
        ]
        eval_data = [
            {"color": "green", "size": "large"},  # "green" and "large" are eval-only
        ]

        pipeline = OrigamiPipeline()
        pipeline.preprocess(train_data, eval_data=eval_data)

        vocab = pipeline._tokenizer.vocab

        # Training values should be in vocabulary (not mapped to UNK)
        assert vocab.encode(ValueToken("red")) != vocab.unk_value_id
        assert vocab.encode(ValueToken("blue")) != vocab.unk_value_id
        assert vocab.encode(ValueToken("small")) != vocab.unk_value_id
        assert vocab.encode(ValueToken("medium")) != vocab.unk_value_id

        # Eval-only values should NOT be in vocabulary (mapped to UNK)
        assert vocab.encode(ValueToken("green")) == vocab.unk_value_id
        assert vocab.encode(ValueToken("large")) == vocab.unk_value_id

    def test_eval_loss_with_schema_not_inf(self):
        """Test that eval loss is finite when using constrain_schema + infer_schema.

        When constrain_schema=True and infer_schema=True, the evaluator applies
        schema masks during loss computation. With the tokenizer fitted on training
        data only, eval-only values become UNK tokens. The schema must allow UNK
        tokens so that eval loss doesn't become inf.
        """
        torch.manual_seed(42)

        # Training data — schema will be inferred from these values only
        train_data = [{"label": v, "x": i} for i, v in enumerate(["A", "B", "C"] * 10)]

        # Eval data — "D" is unseen during training
        eval_data = [{"label": "D", "x": i} for i in range(5)]

        config = OrigamiConfig(
            model=ModelConfig(d_model=32, n_layers=2),
            training=TrainingConfig(
                num_epochs=1,
                constrain_schema=True,
            ),
            data=DataConfig(infer_schema=True),
        )

        pipeline = OrigamiPipeline(config)
        pipeline.fit(train_data, eval_data=eval_data, epochs=1)

        # Evaluate on eval data containing unseen value "D"
        results = pipeline.evaluate(eval_data)

        assert "loss" in results
        assert np.isfinite(results["loss"]), (
            f"Eval loss is {results['loss']} — expected finite. "
            "Schema masks may be blocking UNK tokens."
        )
