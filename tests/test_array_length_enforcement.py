"""Tests for array length enforcement during generation (Phase 3).

Tests that the generator samples array lengths from the continuous head at
ARRAY_START tokens and enforces them via schema mask overrides.
"""

import pytest
import torch

from origami.config import ModelConfig
from origami.constraints.json_grammar import JSONGrammarPDA
from origami.constraints.schema_pda import SchemaPDA
from origami.inference import OrigamiGenerator
from origami.inference.generator import PathState
from origami.model import OrigamiModel
from origami.position_encoding import PATH_TYPE_KEY
from origami.tokenizer import JSONTokenizer
from origami.tokenizer.vocabulary import KeyToken


def _make_model_and_tokenizer(data, seed=42):
    """Helper to create a model and tokenizer with continuous head enabled."""
    torch.manual_seed(seed)
    tokenizer = JSONTokenizer()
    tokenizer.fit(data)

    config = ModelConfig(
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        use_continuous_head=True,
        num_mixture_components=3,
        max_depth=tokenizer.max_depth,
    )
    model = OrigamiModel(config, vocab=tokenizer.vocab)
    model._grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=config.max_depth)
    model.eval()
    return model, tokenizer


class TestArrayLengthSampling:
    """Tests that array lengths are sampled from the continuous head."""

    def test_generator_stores_enforce_flag(self):
        """Generator stores the enforce_array_lengths flag."""
        data = [{"items": [1, 2]}]
        model, tokenizer = _make_model_and_tokenizer(data)

        gen = OrigamiGenerator(model, tokenizer, enforce_array_lengths=True)
        assert gen._enforce_array_lengths is True

        gen2 = OrigamiGenerator(model, tokenizer, enforce_array_lengths=False)
        assert gen2._enforce_array_lengths is False

    def test_enforcement_disabled_without_continuous_head(self):
        """Without continuous head, enforcement silently deactivates."""
        data = [{"items": [1, 2]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)

        config = ModelConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            use_continuous_head=False,
            max_depth=tokenizer.max_depth,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)
        model._grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=config.max_depth)
        model.eval()

        gen = OrigamiGenerator(model, tokenizer, enforce_array_lengths=True)
        # Should still generate without error (enforcement silently disabled)
        results = gen.generate(num_samples=1, max_length=50, seed=42)
        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_generate_with_enforcement_produces_valid_json(self):
        """Generation with enforcement produces valid JSON objects or raises GenerationError.

        With an untrained model, generation may not terminate within max_length
        (especially with deeply nested arrays). Both outcomes are acceptable.
        """
        from origami.inference.utils import GenerationError

        data = [
            {"items": [1, 2, 3], "name": "test"},
            {"items": [4, 5], "name": "other"},
        ]
        model, tokenizer = _make_model_and_tokenizer(data)

        gen = OrigamiGenerator(model, tokenizer, enforce_array_lengths=True)
        try:
            results = gen.generate(num_samples=3, max_length=200, seed=42)
            assert len(results) == 3
            for r in results:
                assert isinstance(r, dict)
        except GenerationError:
            pass  # Expected for untrained models


class TestSchemaArrayLengthMask:
    """Tests for _compute_schema_mask with target length stacks."""

    @pytest.fixture
    def setup(self):
        """Create model, tokenizer, schema, and generator."""
        data = [
            {"items": [1, 2, 3]},
            {"items": [4, 5]},
            {"items": []},
        ]
        model, tokenizer = _make_model_and_tokenizer(data)

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 0,
                    "maxItems": 10,
                },
            },
        }

        gen = OrigamiGenerator(
            model,
            tokenizer,
            constrain_schema=True,
            schema=schema,
            enforce_array_lengths=True,
        )

        return model, tokenizer, gen

    def _make_array_state(self, tokenizer, gen):
        """Create path state and schema state inside an array."""
        vocab = tokenizer.vocab

        ps = PathState()
        ps.push_object()
        key_token = KeyToken("items")
        key_id = vocab.encode(key_token)
        ps.set_key(PATH_TYPE_KEY, key_id)
        ps.push_array()

        ss = gen._schema_pda.init_state_from_tokens(
            [vocab.start_id, vocab.obj_start_id, key_id, vocab.array_start_id],
            vocab,
        )

        return ps, ss

    def test_target_length_suppresses_array_end(self, setup):
        """_compute_schema_mask suppresses ARRAY_END when below target length."""
        _, tokenizer, gen = setup
        vocab = tokenizer.vocab

        ps, ss = self._make_array_state(tokenizer, gen)
        # arr_count should be 0 (no elements yet)

        # target_length_stacks with target of 3
        target_stacks = [[3]]

        mask = gen._compute_schema_mask([ps], [ss], target_stacks)

        # ARRAY_END should be suppressed (0 < 3)
        assert not mask[0, vocab.array_end_id].item()

    def test_target_length_forces_array_end(self, setup):
        """_compute_schema_mask forces ARRAY_END when at target length."""
        _, tokenizer, gen = setup
        vocab = tokenizer.vocab

        ps, ss = self._make_array_state(tokenizer, gen)
        # Simulate 3 elements added
        ss.increment_array_count()
        ss.increment_array_count()
        ss.increment_array_count()

        # target_length_stacks with target of 3 (reached)
        target_stacks = [[3]]

        mask = gen._compute_schema_mask([ps], [ss], target_stacks)

        # Value-like tokens should be suppressed to force ARRAY_END
        assert not mask[0, vocab.obj_start_id].item()
        assert not mask[0, vocab.array_start_id].item()
        assert not mask[0, vocab.num_token_id].item()
        # ARRAY_END should be allowed
        assert mask[0, vocab.array_end_id].item()

    def test_target_length_zero_forces_empty_array(self, setup):
        """Target length 0 forces immediate ARRAY_END (empty array)."""
        _, tokenizer, gen = setup
        vocab = tokenizer.vocab

        ps, ss = self._make_array_state(tokenizer, gen)
        # arr_count = 0, target = 0 → should force ARRAY_END immediately

        target_stacks = [[0]]

        mask = gen._compute_schema_mask([ps], [ss], target_stacks)

        # Value-like tokens should be suppressed
        assert not mask[0, vocab.obj_start_id].item()
        assert not mask[0, vocab.array_start_id].item()
        # ARRAY_END should be allowed
        assert mask[0, vocab.array_end_id].item()

    def test_no_target_falls_back_to_schema(self, setup):
        """Without target length, schema min/max items still apply."""
        _, tokenizer, gen = setup
        vocab = tokenizer.vocab

        ps, ss = self._make_array_state(tokenizer, gen)
        # Simulate 10 elements (at maxItems)
        for _ in range(10):
            ss.increment_array_count()

        # Empty target stack → falls back to schema
        target_stacks = [[]]

        mask = gen._compute_schema_mask([ps], [ss], target_stacks)

        # At maxItems=10, value tokens should be suppressed
        assert not mask[0, vocab.obj_start_id].item()
        assert not mask[0, vocab.array_start_id].item()

    def test_none_target_stacks_falls_back_to_schema(self, setup):
        """None target stacks completely falls back to schema."""
        _, tokenizer, gen = setup
        vocab = tokenizer.vocab

        ps, ss = self._make_array_state(tokenizer, gen)
        for _ in range(10):
            ss.increment_array_count()

        # None → no enforcement at all, pure schema
        mask = gen._compute_schema_mask([ps], [ss], None)

        assert not mask[0, vocab.obj_start_id].item()
        assert not mask[0, vocab.array_start_id].item()


class TestTargetLengthStackManagement:
    """Tests for the target length stack push/pop logic."""

    def test_stack_operations_basic(self):
        """Basic push/pop operations on the target length stack."""
        stack: list[int] = []
        stack.append(3)  # ARRAY_START → push target
        assert stack == [3]
        stack.append(2)  # Nested ARRAY_START → push target
        assert stack == [3, 2]
        stack.pop()  # Inner ARRAY_END → pop
        assert stack == [3]
        stack.pop()  # Outer ARRAY_END → pop
        assert stack == []

    def test_target_stacks_compacted_with_batch(self):
        """Target length stacks are compacted when sequences complete."""
        from origami.inference.utils import GenerationError

        data = [{"items": [1, 2, 3]}]
        model, tokenizer = _make_model_and_tokenizer(data)

        gen = OrigamiGenerator(model, tokenizer, enforce_array_lengths=True)

        # Generate multiple samples in a batch — compaction should not error.
        # Untrained models may not terminate within max_length.
        try:
            results = gen.generate(num_samples=5, batch_size=5, max_length=200, seed=42)
            assert len(results) == 5
        except GenerationError:
            pass  # Expected for untrained models


class TestConfigIntegration:
    """Tests for InferenceConfig integration."""

    def test_inference_config_default(self):
        """InferenceConfig defaults enforce_array_lengths=True."""
        from origami.config import InferenceConfig

        cfg = InferenceConfig()
        assert cfg.enforce_array_lengths is True

    def test_pipeline_passes_enforce_flag(self):
        """Pipeline passes enforce_array_lengths to generator."""
        from origami.config import InferenceConfig, ModelConfig, OrigamiConfig, TrainingConfig
        from origami.pipeline import OrigamiPipeline

        config = OrigamiConfig(
            model=ModelConfig(
                d_model=32,
                n_heads=2,
                n_layers=1,
                d_ff=64,
                use_continuous_head=True,
            ),
            training=TrainingConfig(num_epochs=1, batch_size=2),
            inference=InferenceConfig(enforce_array_lengths=False),
        )
        pipeline = OrigamiPipeline(config)

        data = [
            {"items": [1, 2], "label": "a"},
            {"items": [3], "label": "b"},
        ]
        pipeline.fit(data)

        generator = pipeline._get_generator()
        assert generator._enforce_array_lengths is False

    def test_predictor_passes_enforce_flag(self):
        """Predictor passes enforce_array_lengths to its internal generator."""
        from origami.inference import OrigamiPredictor

        data = [{"items": [1, 2]}]
        model, tokenizer = _make_model_and_tokenizer(data)

        predictor = OrigamiPredictor(
            model, tokenizer, enforce_array_lengths=False
        )
        assert predictor._generator._enforce_array_lengths is False

    def test_evaluator_passes_enforce_flag(self):
        """Evaluator passes enforce_array_lengths to predictor."""
        from origami.inference.evaluator import OrigamiEvaluator

        data = [{"items": [1, 2]}]
        model, tokenizer = _make_model_and_tokenizer(data)

        evaluator = OrigamiEvaluator(
            model, tokenizer, enforce_array_lengths=False
        )
        assert evaluator._enforce_array_lengths is False
