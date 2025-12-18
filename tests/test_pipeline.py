"""Tests for OrigamiPipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from origami.pipeline import OrigamiPipeline, PipelineConfig


class TestPipelineConfig:
    """Tests for PipelineConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 4
        assert config.numeric_mode == "none"
        assert config.batch_size == 32

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            d_model=256,
            n_heads=8,
            numeric_mode="scale",
            cat_threshold=50,
        )
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.numeric_mode == "scale"
        assert config.cat_threshold == 50

    def test_config_validation_d_model_divisible_by_n_heads(self):
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValueError, match="divisible"):
            PipelineConfig(d_model=100, n_heads=3)

    def test_config_validation_n_layers(self):
        """Test that n_layers must be >= 1."""
        with pytest.raises(ValueError, match="n_layers"):
            PipelineConfig(n_layers=0)

    def test_config_validation_cat_threshold(self):
        """Test that cat_threshold must be >= 1."""
        with pytest.raises(ValueError, match="cat_threshold"):
            PipelineConfig(cat_threshold=0)

    def test_config_validation_n_bins(self):
        """Test that n_bins must be >= 2 for discretize mode."""
        with pytest.raises(ValueError, match="n_bins"):
            PipelineConfig(numeric_mode="discretize", n_bins=1)


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
        assert pipeline._preprocessor is None  # numeric_mode="none"

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

        config = PipelineConfig(
            numeric_mode="discretize",
            cat_threshold=10,
            n_bins=5,
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

        config = PipelineConfig(
            numeric_mode="scale",
            cat_threshold=10,
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
            assert loaded.config.d_model == fitted_pipeline.config.d_model
        finally:
            path.unlink()

    def test_save_load_with_scaler(self):
        """Test save/load with NumericScaler."""
        data = [{"x": float(i), "y": i % 3} for i in range(200)]

        config = PipelineConfig(numeric_mode="scale", cat_threshold=10)
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            pipeline.save(path)
            loaded = OrigamiPipeline.load(path)

            assert loaded.config.numeric_mode == "scale"
            assert loaded._preprocessor is not None
            from origami.preprocessing import NumericScaler

            assert isinstance(loaded._preprocessor, NumericScaler)
        finally:
            path.unlink()

    def test_save_load_with_discretizer(self):
        """Test save/load with NumericDiscretizer."""
        data = [{"x": float(i), "y": i % 3} for i in range(200)]

        config = PipelineConfig(numeric_mode="discretize", cat_threshold=10, n_bins=5)
        pipeline = OrigamiPipeline(config)
        pipeline.fit(data, epochs=1, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)

        try:
            pipeline.save(path)
            loaded = OrigamiPipeline.load(path)

            assert loaded.config.numeric_mode == "discretize"
            assert loaded._preprocessor is not None
            from origami.preprocessing import NumericDiscretizer

            assert isinstance(loaded._preprocessor, NumericDiscretizer)
        finally:
            path.unlink()

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
        config = PipelineConfig(batch_size=2, d_model=32, n_heads=2, n_layers=2)
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

        config = PipelineConfig(d_model=16, n_heads=4, n_layers=4)
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
        samples1 = fitted_pipeline.generate(num_samples=2, seed=42)
        samples2 = fitted_pipeline.generate(num_samples=2, seed=42)

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
        config = PipelineConfig(d_model=32, n_heads=4, n_layers=2)
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
        assert "none" in repr(pipeline)

    def test_repr_fitted(self):
        """Test repr for fitted pipeline."""
        data = [{"a": 1}, {"a": 2}]
        pipeline = OrigamiPipeline()
        pipeline.fit(data, epochs=1, verbose=False)
        assert "fitted" in repr(pipeline)
        assert "not fitted" not in repr(pipeline)

    def test_repr_with_scale_mode(self):
        """Test repr shows numeric mode."""
        config = PipelineConfig(numeric_mode="scale")
        pipeline = OrigamiPipeline(config)
        assert "scale" in repr(pipeline)


class TestPipelineDeviceManagement:
    """Tests for automatic device management."""

    def test_model_on_cpu_after_inference(self):
        """Test that model moves to CPU after first inference call."""
        data = [{"a": i, "b": i * 2} for i in range(50)]

        # Force CPU device in config to test the logic
        config = PipelineConfig(d_model=32, n_layers=2, device="cpu")
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
        config = PipelineConfig(d_model=32, n_layers=2, device="cpu")
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
        config = PipelineConfig(d_model=32, n_layers=2, device="cpu")
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

        config = PipelineConfig(d_model=32, n_layers=2, device="cpu")
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
