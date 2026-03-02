"""Tests for sequential positional encoding (ablation alternative to KVPE)."""

import pytest
import torch

from origami.config import ModelConfig, OrigamiConfig
from origami.model import OrigamiEmbeddings, OrigamiModel
from origami.tokenizer import JSONTokenizer
from origami.training import OrigamiDataCollator
from origami.utils import available_devices as get_available_devices

AVAILABLE_DEVICES = get_available_devices()


# --- Shared fixtures ---


@pytest.fixture
def tokenizer():
    """Create a tokenizer fitted on sample data."""
    tokenizer = JSONTokenizer()
    tokenizer.fit([
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ])
    return tokenizer


@pytest.fixture
def seq_config():
    """ModelConfig with sequential PE."""
    return ModelConfig(
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        position_encoding="sequential",
    )


# --- Config tests ---


class TestModelConfigPositionEncoding:
    """Config validation for position_encoding field."""

    def test_default_is_kvpe(self):
        config = ModelConfig()
        assert config.position_encoding == "kvpe"

    def test_sequential_valid(self):
        config = ModelConfig(position_encoding="sequential")
        assert config.position_encoding == "sequential"

    def test_invalid_position_encoding_raises(self):
        with pytest.raises(ValueError, match="position_encoding"):
            ModelConfig(position_encoding="sinusoidal")

    def test_kvpe_fields_accepted_when_sequential(self):
        """KVPE-specific fields should not cause errors when sequential."""
        config = ModelConfig(
            position_encoding="sequential",
            kvpe_pooling="gru",
            kvpe_pooling_kwargs={"num_layers": 2},
            max_depth=16,
        )
        assert config.position_encoding == "sequential"


# --- Embedding unit tests ---


class TestSequentialEmbeddings:
    """Tests for OrigamiEmbeddings with sequential PE."""

    VOCAB_SIZE = 100

    def test_has_position_embedding(self, seq_config):
        """Sequential PE should create nn.Embedding, not KVPE."""
        emb = OrigamiEmbeddings(seq_config, self.VOCAB_SIZE)
        assert hasattr(emb, "position_embedding")
        assert not hasattr(emb, "kvpe")

    def test_kvpe_has_no_position_embedding(self):
        """KVPE mode should not create position_embedding."""
        config = ModelConfig(d_model=64, n_heads=4, position_encoding="kvpe")
        emb = OrigamiEmbeddings(config, self.VOCAB_SIZE)
        assert hasattr(emb, "kvpe")
        assert not hasattr(emb, "position_embedding")

    def test_forward_shape(self, seq_config):
        batch_size, seq_len = 2, 10
        emb = OrigamiEmbeddings(seq_config, self.VOCAB_SIZE)
        input_ids = torch.randint(0, self.VOCAB_SIZE, (batch_size, seq_len))
        path_types = torch.zeros(batch_size, seq_len, seq_config.max_depth, dtype=torch.long)
        path_ids = torch.zeros(batch_size, seq_len, seq_config.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        output = emb(
            input_ids, path_types, path_ids, path_lengths,
            position_ids=position_ids,
        )
        assert output.shape == (batch_size, seq_len, seq_config.d_model)

    def test_different_positions_different_embeddings(self, seq_config):
        """Different position IDs should produce different embeddings."""
        emb = OrigamiEmbeddings(seq_config, self.VOCAB_SIZE)
        # Same token at two different positions
        input_ids = torch.tensor([[5, 5]])
        path_types = torch.zeros(1, 2, seq_config.max_depth, dtype=torch.long)
        path_ids = torch.zeros(1, 2, seq_config.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(1, 2, dtype=torch.long)
        position_ids = torch.tensor([[0, 1]])

        output = emb(
            input_ids, path_types, path_ids, path_lengths,
            position_ids=position_ids,
        )
        # Position 0 and position 1 should produce different embeddings
        assert not torch.allclose(output[0, 0], output[0, 1])

    def test_same_position_same_embedding(self, seq_config):
        """Same token at same position should produce identical embeddings."""
        emb = OrigamiEmbeddings(seq_config, self.VOCAB_SIZE)
        input_ids = torch.tensor([[5, 5]])
        path_types = torch.zeros(1, 2, seq_config.max_depth, dtype=torch.long)
        path_ids = torch.zeros(1, 2, seq_config.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(1, 2, dtype=torch.long)
        # Same position ID for both tokens
        position_ids = torch.tensor([[3, 3]])

        output = emb(
            input_ids, path_types, path_ids, path_lengths,
            position_ids=position_ids,
        )
        assert torch.allclose(output[0, 0], output[0, 1])

    def test_max_seq_length_respected(self, seq_config):
        """position_embedding should have max_seq_length entries."""
        emb = OrigamiEmbeddings(seq_config, self.VOCAB_SIZE)
        assert emb.position_embedding.num_embeddings == seq_config.max_seq_length

    def test_gradient_flow(self, seq_config):
        """Gradients should flow through position embeddings."""
        emb = OrigamiEmbeddings(seq_config, self.VOCAB_SIZE)
        input_ids = torch.randint(0, self.VOCAB_SIZE, (1, 5))
        path_types = torch.zeros(1, 5, seq_config.max_depth, dtype=torch.long)
        path_ids = torch.zeros(1, 5, seq_config.max_depth, dtype=torch.long)
        path_lengths = torch.zeros(1, 5, dtype=torch.long)
        position_ids = torch.arange(5).unsqueeze(0)

        output = emb(
            input_ids, path_types, path_ids, path_lengths,
            position_ids=position_ids,
        )
        output.sum().backward()

        assert emb.position_embedding.weight.grad is not None
        assert emb.position_embedding.weight.grad.any()

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_forward_on_device(self, seq_config, device):
        emb = OrigamiEmbeddings(seq_config, self.VOCAB_SIZE).to(device)
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.VOCAB_SIZE, (batch_size, seq_len), device=device)
        path_types = torch.zeros(
            batch_size, seq_len, seq_config.max_depth, dtype=torch.long, device=device,
        )
        path_ids = torch.zeros(
            batch_size, seq_len, seq_config.max_depth, dtype=torch.long, device=device,
        )
        path_lengths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        output = emb(
            input_ids, path_types, path_ids, path_lengths,
            position_ids=position_ids,
        )
        assert output.device.type == device.type
        assert output.shape == (batch_size, seq_len, seq_config.d_model)


# --- Model integration tests ---


class TestSequentialModel:
    """Integration tests for OrigamiModel with sequential PE."""

    def test_forward_no_labels(self, seq_config, tokenizer):
        model = OrigamiModel(seq_config, vocab=tokenizer.vocab)
        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        batch = collator.collate_objects([{"name": "Alice", "age": 30}])
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
        )
        assert output.loss is None
        assert output.logits.shape == (1, batch.input_ids.size(1), tokenizer.vocab.size)
        assert not torch.isnan(output.logits).any()

    def test_forward_with_labels(self, seq_config, tokenizer):
        model = OrigamiModel(seq_config, vocab=tokenizer.vocab)
        collator = OrigamiDataCollator(tokenizer, include_labels=True)
        batch = collator.collate_objects([{"name": "Alice", "age": 30}])
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        assert output.loss is not None
        assert output.loss.ndim == 0
        assert torch.isfinite(output.loss)

    def test_forward_batch_with_padding(self, seq_config, tokenizer):
        """Left-padded batches should work with sequential PE."""
        model = OrigamiModel(seq_config, vocab=tokenizer.vocab)
        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        # Different-length objects create left-padding
        objects = [
            {"name": "Alice", "age": 30},
            {"name": "Bob"},
        ]
        batch = collator.collate_objects(objects)
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
        )
        assert output.logits.shape[0] == 2
        assert not torch.isnan(output.logits).any()

    def test_position_ids_from_attention_mask(self):
        """Verify position IDs are correctly derived from attention_mask."""
        attention_mask = torch.tensor([
            [False, False, True, True, True],  # 2 PAD + 3 real
            [True, True, True, True, True],    # 5 real
        ])
        expected = torch.tensor([
            [0, 0, 0, 1, 2],
            [0, 1, 2, 3, 4],
        ])
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        assert torch.equal(position_ids, expected)

    def test_position_ids_without_attention_mask(self, seq_config, tokenizer):
        """Without attention_mask, should use simple sequential positions."""
        model = OrigamiModel(seq_config, vocab=tokenizer.vocab)
        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        batch = collator.collate_objects([{"name": "Alice"}])
        # Call without attention_mask
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
        )
        assert not torch.isnan(output.logits).any()

    def test_explicit_position_ids(self, seq_config, tokenizer):
        """Explicitly provided position_ids should be used directly."""
        model = OrigamiModel(seq_config, vocab=tokenizer.vocab)
        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        batch = collator.collate_objects([{"name": "Alice"}])
        seq_len = batch.input_ids.size(1)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            position_ids=position_ids,
        )
        assert not torch.isnan(output.logits).any()

    def test_gradient_flow_to_position_embedding(self, seq_config, tokenizer):
        model = OrigamiModel(seq_config, vocab=tokenizer.vocab)
        collator = OrigamiDataCollator(tokenizer, include_labels=True)
        batch = collator.collate_objects([{"name": "Alice", "age": 30}])
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        output.loss.backward()
        assert model.embeddings.position_embedding.weight.grad is not None

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_forward_on_device(self, seq_config, tokenizer, device):
        model = OrigamiModel(seq_config, vocab=tokenizer.vocab).to(device)
        collator = OrigamiDataCollator(tokenizer, include_labels=True)
        batch = collator.collate_objects([{"name": "Alice", "age": 30}])
        batch = batch.to(device)
        output = model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        assert output.logits.device.type == device.type
        assert output.loss.device.type == device.type


# --- KV cache tests ---


class TestSequentialPEKVCache:
    """Test KV cache works correctly with sequential PE."""

    @pytest.fixture
    def cached_config(self):
        return ModelConfig(
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            position_encoding="sequential",
            backbone="cached_transformer",
        )

    def test_kv_cache_forward(self, cached_config, tokenizer):
        """KV-cached forward should work with sequential PE."""
        model = OrigamiModel(cached_config, vocab=tokenizer.vocab)
        model.eval()

        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        batch = collator.collate_objects([{"name": "Alice", "age": 30}])

        # First forward: full sequence with cache
        with torch.no_grad():
            output1 = model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                use_cache=True,
            )
        kv_cache = output1.past_key_values
        assert kv_cache is not None

        # Second forward: single new token with KV cache
        new_token = torch.tensor([[tokenizer.vocab.obj_end_id]])
        max_depth = batch.path_types.size(2)
        new_path_types = torch.zeros(1, 1, max_depth, dtype=torch.long)
        new_path_ids = torch.zeros(1, 1, max_depth, dtype=torch.long)
        new_path_lengths = torch.zeros(1, 1, dtype=torch.long)

        # Compute correct position_ids for the new token
        past_real_tokens = batch.attention_mask.long().sum()
        position_ids = torch.tensor([[past_real_tokens]])

        with torch.no_grad():
            output2 = model(
                input_ids=new_token,
                path_types=new_path_types,
                path_ids=new_path_ids,
                path_lengths=new_path_lengths,
                past_key_values=kv_cache,
                use_cache=True,
                position_ids=position_ids,
            )
        assert output2.logits.shape == (1, 1, tokenizer.vocab.size)
        assert not torch.isnan(output2.logits).any()


# --- Generation and prediction tests ---


class TestSequentialPEGeneration:
    """Test generation and prediction with sequential PE."""

    @pytest.fixture
    def small_tokenizer(self):
        tokenizer = JSONTokenizer()
        tokenizer.fit([
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ])
        return tokenizer

    def test_generate_produces_valid_objects(self, small_tokenizer):
        """Generation should produce valid JSON objects."""
        from origami.inference import OrigamiGenerator

        config = ModelConfig(
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            position_encoding="sequential",
        )
        model = OrigamiModel(config, vocab=small_tokenizer.vocab)
        generator = OrigamiGenerator(model, small_tokenizer, constrain_grammar=True)
        results = generator.generate(num_samples=3, max_length=200, temperature=1.0)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_predict_runs(self, small_tokenizer):
        """Prediction should work with sequential PE."""
        from origami.inference import OrigamiPredictor

        config = ModelConfig(
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            position_encoding="sequential",
        )
        model = OrigamiModel(config, vocab=small_tokenizer.vocab)
        predictor = OrigamiPredictor(model, small_tokenizer, constrain_grammar=True)
        result = predictor.predict({"a": 1, "b": None}, target_key="b")
        assert result is not None

    def test_predict_proba_runs(self, small_tokenizer):
        """predict_proba should work with sequential PE."""
        from origami.inference import OrigamiPredictor

        config = ModelConfig(
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            position_encoding="sequential",
        )
        model = OrigamiModel(config, vocab=small_tokenizer.vocab)
        predictor = OrigamiPredictor(model, small_tokenizer, constrain_grammar=True)
        probs = predictor.predict_proba({"a": 1, "b": None}, target_key="b", top_k=3)
        assert isinstance(probs, list)
        assert len(probs) > 0
        # Each entry is (value, probability)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in probs)

    def test_generate_with_kv_cache(self, small_tokenizer):
        """Generation with cached_transformer backbone should work."""
        from origami.inference import OrigamiGenerator

        config = ModelConfig(
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            position_encoding="sequential",
            backbone="cached_transformer",
        )
        model = OrigamiModel(config, vocab=small_tokenizer.vocab)
        generator = OrigamiGenerator(model, small_tokenizer, constrain_grammar=True)
        results = generator.generate(num_samples=2, max_length=50, temperature=1.0)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)


# --- Save/Load tests ---


class TestSequentialPESaveLoad:
    """Save/load roundtrip preserves config and weights."""

    def test_model_save_load_roundtrip(self, tmp_path, tokenizer):
        config = ModelConfig(
            d_model=32, n_heads=2, n_layers=1,
            position_encoding="sequential",
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        path = tmp_path / "model.pt"
        model.save(path, tokenizer)

        loaded_model, loaded_tokenizer = OrigamiModel.load(path)
        assert loaded_model.config.position_encoding == "sequential"
        assert hasattr(loaded_model.embeddings, "position_embedding")
        assert not hasattr(loaded_model.embeddings, "kvpe")

        # Weights should match
        for key in model.state_dict():
            assert torch.allclose(
                model.state_dict()[key],
                loaded_model.state_dict()[key],
            ), f"Weight mismatch for {key}"

    def test_pipeline_save_load_roundtrip(self, tmp_path):
        from origami.pipeline import OrigamiPipeline

        config = OrigamiConfig(
            model=ModelConfig(
                d_model=32, n_heads=2, n_layers=1,
                position_encoding="sequential",
            ),
        )
        pipeline = OrigamiPipeline(config)
        data = [{"a": 1, "b": 2}] * 10
        pipeline.fit(data, epochs=1, callbacks=[])

        path = tmp_path / "pipeline.pt"
        pipeline.save(path)

        loaded = OrigamiPipeline.load(path)
        assert loaded.config.model.position_encoding == "sequential"

    def test_backward_compat_old_checkpoint(self, tmp_path, tokenizer):
        """Loading a checkpoint without position_encoding should default to kvpe."""
        config = ModelConfig(d_model=32, n_heads=2, n_layers=1)
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        path = tmp_path / "model.pt"
        model.save(path, tokenizer)

        # Simulate old checkpoint by removing the field
        checkpoint = torch.load(path, weights_only=False)
        del checkpoint["model_config"]["position_encoding"]
        torch.save(checkpoint, path)

        loaded_model, _ = OrigamiModel.load(path)
        assert loaded_model.config.position_encoding == "kvpe"
        assert hasattr(loaded_model.embeddings, "kvpe")


# --- Comparison test ---


class TestKVPEvsSequentialComparison:
    """Sanity check that KVPE and sequential produce different embeddings."""

    def test_different_hidden_states(self, tokenizer):
        config_kvpe = ModelConfig(
            d_model=64, n_heads=4, n_layers=1, d_ff=128,
            position_encoding="kvpe",
        )
        config_seq = ModelConfig(
            d_model=64, n_heads=4, n_layers=1, d_ff=128,
            position_encoding="sequential",
        )

        # Use same seed for token embeddings + backbone weights
        torch.manual_seed(42)
        model_kvpe = OrigamiModel(config_kvpe, vocab=tokenizer.vocab)
        torch.manual_seed(42)
        model_seq = OrigamiModel(config_seq, vocab=tokenizer.vocab)

        collator = OrigamiDataCollator(tokenizer, include_labels=False)
        batch = collator.collate_objects([{"name": "Alice", "age": 30}])

        with torch.no_grad():
            out_kvpe = model_kvpe(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
            )
            out_seq = model_seq(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
            )

        # Hidden states should differ due to different position encodings
        assert not torch.allclose(out_kvpe.hidden_states, out_seq.hidden_states)
