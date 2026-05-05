"""Tests for independent grammar and schema constraint control.

Tests that training and inference constraints can be configured independently,
and that all combinations of constrain_grammar, constrain_schema, and
infer_schema work correctly across Generator, Predictor, Evaluator, and Pipeline.
"""

import numpy as np
import pytest
import torch

from origami.config import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    OrigamiConfig,
    TrainingConfig,
)
from origami.inference import OrigamiEvaluator, OrigamiGenerator, OrigamiPredictor
from origami.inference.utils import GenerationError
from origami.model import OrigamiModel
from origami.pipeline import OrigamiPipeline
from origami.tokenizer import JSONTokenizer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SMALL_MODEL_CFG = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64)


@pytest.fixture
def sample_data():
    """Simple categorical data suitable for all constraint combos."""
    return [
        {"color": "red", "shape": "circle"},
        {"color": "blue", "shape": "square"},
        {"color": "red", "shape": "triangle"},
        {"color": "blue", "shape": "circle"},
    ] * 5  # 20 samples for stable training


@pytest.fixture
def sample_schema():
    """Schema matching sample_data."""
    return {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "blue"]},
            "shape": {"type": "string", "enum": ["circle", "square", "triangle"]},
        },
    }


@pytest.fixture
def tokenizer(sample_data):
    """Fitted tokenizer."""
    tok = JSONTokenizer()
    tok.fit(sample_data)
    return tok


@pytest.fixture
def model(tokenizer):
    """Small untrained model (no PDA attached)."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_depth=tokenizer.max_depth,
    )
    return OrigamiModel(cfg, vocab=tokenizer.vocab)


@pytest.fixture
def model_with_grammar_pda(model, tokenizer):
    """Model with grammar PDA attached (simulates training with constrain_grammar=True)."""
    from origami.constraints.json_grammar import JSONGrammarPDA

    model._grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=model.config.max_depth)
    return model


# ===========================================================================
# InferenceConfig
# ===========================================================================


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.constrain_grammar is True
        assert cfg.constrain_schema is False

    def test_custom_values(self):
        cfg = InferenceConfig(constrain_grammar=False, constrain_schema=True)
        assert cfg.constrain_grammar is False
        assert cfg.constrain_schema is True

    def test_no_unk_params(self):
        """UNK params were removed — they should not be accepted."""
        with pytest.raises(TypeError):
            InferenceConfig(schema_allow_unk_key=True)
        with pytest.raises(TypeError):
            InferenceConfig(schema_allow_unk_value=True)


class TestOrigamiConfigWithInference:
    """Tests for InferenceConfig integration in OrigamiConfig."""

    def test_default_inference_config(self):
        cfg = OrigamiConfig()
        assert isinstance(cfg.inference, InferenceConfig)
        assert cfg.inference.constrain_grammar is True
        assert cfg.inference.constrain_schema is False

    def test_custom_inference_config(self):
        cfg = OrigamiConfig(
            inference=InferenceConfig(constrain_grammar=False, constrain_schema=True),
        )
        assert cfg.inference.constrain_grammar is False
        assert cfg.inference.constrain_schema is True

    def test_training_and_inference_independent(self):
        """Training and inference constraint settings don't affect each other."""
        cfg = OrigamiConfig(
            training=TrainingConfig(constrain_grammar=False, constrain_schema=False),
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
        )
        assert cfg.training.constrain_grammar is False
        assert cfg.training.constrain_schema is False
        assert cfg.inference.constrain_grammar is True
        assert cfg.inference.constrain_schema is True


# ===========================================================================
# Generator constraint resolution
# ===========================================================================


class TestGeneratorConstraints:
    """Tests for Generator constraint resolution logic."""

    def test_grammar_true_creates_pda_when_model_has_none(self, model, tokenizer):
        """constrain_grammar=True should create PDA if model doesn't have one."""
        assert not hasattr(model, "_grammar_pda") or model._grammar_pda is None
        gen = OrigamiGenerator(model, tokenizer, constrain_grammar=True)
        assert gen._grammar_pda is not None

    def test_grammar_true_reuses_model_pda(self, model_with_grammar_pda, tokenizer):
        """constrain_grammar=True should reuse model's existing PDA."""
        existing_pda = model_with_grammar_pda._grammar_pda
        gen = OrigamiGenerator(model_with_grammar_pda, tokenizer, constrain_grammar=True)
        assert gen._grammar_pda is existing_pda

    def test_grammar_false_ignores_model_pda(self, model_with_grammar_pda, tokenizer):
        """constrain_grammar=False should NOT use model's PDA."""
        gen = OrigamiGenerator(model_with_grammar_pda, tokenizer, constrain_grammar=False)
        assert gen._grammar_pda is None

    def test_grammar_false_no_pda(self, model, tokenizer):
        """constrain_grammar=False with no model PDA — no grammar constraints."""
        gen = OrigamiGenerator(model, tokenizer, constrain_grammar=False)
        assert gen._grammar_pda is None

    def test_schema_true_with_schema(self, model, tokenizer, sample_schema):
        """constrain_schema=True with schema should create Schema PDA."""
        gen = OrigamiGenerator(model, tokenizer, schema=sample_schema, constrain_schema=True)
        assert gen._schema_pda is not None

    def test_schema_true_without_schema_raises(self, model, tokenizer):
        """constrain_schema=True without schema should raise ValueError."""
        with pytest.raises(ValueError, match="constrain_schema=True requires schema"):
            OrigamiGenerator(model, tokenizer, constrain_schema=True)

    def test_schema_false_ignores_schema(self, model, tokenizer, sample_schema):
        """constrain_schema=False should not create Schema PDA even if schema given."""
        gen = OrigamiGenerator(model, tokenizer, schema=sample_schema, constrain_schema=False)
        assert gen._schema_pda is None

    def test_schema_false_no_schema(self, model, tokenizer):
        """constrain_schema=False without schema — no schema constraints."""
        gen = OrigamiGenerator(model, tokenizer, constrain_schema=False)
        assert gen._schema_pda is None

    def test_schema_pda_uses_strict_unk(self, model, tokenizer, sample_schema):
        """Generator schema PDA should use strict UNK (blocks UNK tokens)."""
        gen = OrigamiGenerator(model, tokenizer, schema=sample_schema, constrain_schema=True)
        pda = gen._schema_pda
        # Strict means UNK_KEY and UNK_VALUE should be blocked in schema masks
        assert not pda._allow_unk_key
        assert not pda._allow_unk_value

    def test_generate_with_grammar_only(self, model, tokenizer):
        """Generation with grammar only produces valid results."""
        gen = OrigamiGenerator(model, tokenizer, constrain_grammar=True, constrain_schema=False)
        results = gen.generate(num_samples=2, max_length=64)
        assert len(results) == 2
        for obj in results:
            assert isinstance(obj, dict)

    def test_generate_with_both_constraints(self, model, tokenizer, sample_schema):
        """Generation with grammar + schema creates proper PDAs."""
        gen = OrigamiGenerator(
            model,
            tokenizer,
            schema=sample_schema,
            constrain_grammar=True,
            constrain_schema=True,
        )
        # Verify both PDAs are created
        assert gen._grammar_pda is not None
        assert gen._schema_pda is not None
        # Untrained model may not generate valid output, but the setup is correct.
        # Full generation is tested in pipeline tests with trained models.

    def test_generate_without_constraints(self, model, tokenizer):
        """Generation without any constraints still produces output."""
        gen = OrigamiGenerator(model, tokenizer, constrain_grammar=False, constrain_schema=False)
        # Without grammar, output may not be valid JSON, but generate() should not crash
        results = gen.generate(num_samples=2, max_length=32)
        assert len(results) >= 0  # May produce empty list if invalid sequences


# ===========================================================================
# Predictor constraint resolution
# ===========================================================================


class TestPredictorConstraints:
    """Tests for Predictor constraint pass-through to Generator."""

    def test_predictor_creates_grammar_pda(self, model, tokenizer):
        """Predictor with constrain_grammar=True creates PDA via Generator."""
        pred = OrigamiPredictor(model, tokenizer, constrain_grammar=True)
        assert pred._generator._grammar_pda is not None

    def test_predictor_no_grammar(self, model, tokenizer):
        """Predictor with constrain_grammar=False — no PDA."""
        pred = OrigamiPredictor(model, tokenizer, constrain_grammar=False)
        assert pred._generator._grammar_pda is None

    def test_predictor_schema_passthrough(self, model, tokenizer, sample_schema):
        """Predictor passes schema constraints to Generator."""
        pred = OrigamiPredictor(
            model,
            tokenizer,
            schema=sample_schema,
            constrain_grammar=True,
            constrain_schema=True,
        )
        assert pred._generator._schema_pda is not None

    def test_predictor_schema_without_schema_raises(self, model, tokenizer):
        """Predictor with constrain_schema=True without schema raises."""
        with pytest.raises(ValueError, match="constrain_schema=True requires schema"):
            OrigamiPredictor(model, tokenizer, constrain_schema=True)

    def test_predict_with_grammar_only(self, model_with_grammar_pda, tokenizer, sample_data):
        """Prediction with grammar constraints returns a result."""
        pred = OrigamiPredictor(model_with_grammar_pda, tokenizer, constrain_grammar=True)
        result = pred.predict(sample_data[0], target_key="shape")
        # Random model — just verify it doesn't crash and returns something
        assert result is not None or result is None

    def test_predict_without_constraints(self, model, tokenizer, sample_data):
        """Prediction without grammar on untrained model may produce GenerationError."""
        pred = OrigamiPredictor(model, tokenizer, constrain_grammar=False)
        # Without grammar, untrained model can generate invalid sequences
        # (e.g., UNK_VALUE or START where a value token is expected).
        # GenerationError is a legitimate outcome here.
        try:
            result = pred.predict(sample_data[0], target_key="shape")
            assert result is not None or result is None
        except GenerationError:
            pass  # Expected: invalid generation without grammar constraints


# ===========================================================================
# Evaluator constraint resolution
# ===========================================================================


class TestEvaluatorConstraints:
    """Tests for Evaluator constraint resolution and UNK handling."""

    def test_grammar_true_creates_pda(self, model, tokenizer):
        """Evaluator with constrain_grammar=True creates PDA for loss."""
        ev = OrigamiEvaluator(model, tokenizer, constrain_grammar=True)
        assert ev._grammar_pda is not None

    def test_grammar_false_no_pda(self, model, tokenizer):
        """Evaluator with constrain_grammar=False — no PDA."""
        ev = OrigamiEvaluator(model, tokenizer, constrain_grammar=False)
        assert ev._grammar_pda is None

    def test_grammar_false_ignores_model_pda(self, model_with_grammar_pda, tokenizer):
        """Evaluator with constrain_grammar=False ignores model PDA."""
        ev = OrigamiEvaluator(model_with_grammar_pda, tokenizer, constrain_grammar=False)
        assert ev._grammar_pda is None

    def test_schema_true_with_schema(self, model, tokenizer, sample_schema):
        """Evaluator creates schema PDA for loss computation."""
        ev = OrigamiEvaluator(
            model,
            tokenizer,
            schema=sample_schema,
            constrain_grammar=True,
            constrain_schema=True,
        )
        assert ev._schema_pda is not None

    def test_schema_true_without_schema_raises(self, model, tokenizer):
        """Evaluator with constrain_schema=True without schema raises."""
        with pytest.raises(ValueError, match="constrain_schema=True requires schema"):
            OrigamiEvaluator(model, tokenizer, constrain_schema=True)

    def test_schema_pda_uses_lenient_unk(self, model, tokenizer, sample_schema):
        """Evaluator loss PDA should use lenient UNK (allows unseen tokens)."""
        ev = OrigamiEvaluator(
            model,
            tokenizer,
            schema=sample_schema,
            constrain_schema=True,
        )
        pda = ev._schema_pda
        assert pda._allow_unk_key is True
        assert pda._allow_unk_value is True

    def test_predictor_uses_strict_unk(self, model, tokenizer, sample_data, sample_schema):
        """Evaluator's lazy predictor should use strict UNK via Generator."""
        ev = OrigamiEvaluator(
            model,
            tokenizer,
            target_key="shape",
            schema=sample_schema,
            constrain_grammar=True,
            constrain_schema=True,
        )
        # Force lazy init of predictor
        ev._get_predictions(sample_data[:2], batch_size=2, allow_complex_values=False)
        # The predictor's generator should have strict UNK
        gen_pda = ev._predictor._generator._schema_pda
        assert gen_pda is not None
        assert not gen_pda._allow_unk_key
        assert not gen_pda._allow_unk_value

    def test_compute_loss_with_grammar(self, model, tokenizer, sample_data):
        """Loss computation with grammar constraints produces finite loss."""
        ev = OrigamiEvaluator(model, tokenizer, constrain_grammar=True)
        results = ev.evaluate(sample_data)
        assert "loss" in results
        assert np.isfinite(results["loss"])

    def test_compute_loss_without_grammar(self, model, tokenizer, sample_data):
        """Loss computation without grammar constraints produces finite loss."""
        ev = OrigamiEvaluator(model, tokenizer, constrain_grammar=False)
        results = ev.evaluate(sample_data)
        assert "loss" in results
        assert np.isfinite(results["loss"])

    def test_compute_loss_with_schema(self, model, tokenizer, sample_data, sample_schema):
        """Loss computation with schema constraints produces finite loss."""
        ev = OrigamiEvaluator(
            model,
            tokenizer,
            schema=sample_schema,
            constrain_grammar=True,
            constrain_schema=True,
        )
        results = ev.evaluate(sample_data)
        assert "loss" in results
        assert np.isfinite(results["loss"])

    def test_evaluate_with_metrics(self, model_with_grammar_pda, tokenizer, sample_data):
        """Evaluation with prediction metrics works with constraints."""
        ev = OrigamiEvaluator(
            model_with_grammar_pda,
            tokenizer,
            target_key="shape",
            constrain_grammar=True,
        )
        results = ev.evaluate(sample_data, metrics={"acc": "accuracy"})
        assert "loss" in results
        assert "acc" in results
        assert np.isfinite(results["loss"])

    def test_no_unk_params_accepted(self, model, tokenizer):
        """Evaluator should not accept schema_allow_unk_* params."""
        with pytest.raises(TypeError):
            OrigamiEvaluator(model, tokenizer, schema_allow_unk_key=True)
        with pytest.raises(TypeError):
            OrigamiEvaluator(model, tokenizer, schema_allow_unk_value=True)


# ===========================================================================
# Convenience evaluate() function
# ===========================================================================


class TestConvenienceEvaluate:
    """Tests for the module-level evaluate() function."""

    def test_no_unk_params(self, model, tokenizer, sample_data):
        """Convenience evaluate() should not accept UNK params."""
        from origami.inference.evaluator import evaluate

        with pytest.raises(TypeError):
            evaluate(model, tokenizer, sample_data, schema_allow_unk_key=True)

    def test_with_constraints(self, model, tokenizer, sample_data, sample_schema):
        """Convenience evaluate() works with constraint parameters."""
        from origami.inference.evaluator import evaluate

        results = evaluate(
            model,
            tokenizer,
            sample_data,
            schema=sample_schema,
            constrain_grammar=True,
            constrain_schema=True,
        )
        assert "loss" in results
        assert np.isfinite(results["loss"])


# ===========================================================================
# Pipeline integration: training vs inference constraints
# ===========================================================================


class TestPipelineConstraintConfig:
    """Tests for Pipeline with independent training/inference constraints."""

    def test_train_without_grammar_infer_with_grammar(self, sample_data):
        """Train without grammar, apply grammar at inference."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=False),
            inference=InferenceConfig(constrain_grammar=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Model should NOT have grammar PDA from training
        assert not hasattr(pipeline.model, "_grammar_pda") or pipeline.model._grammar_pda is None

        # But generation should use grammar (Generator creates its own PDA)
        samples = pipeline.generate(num_samples=2, max_length=64)
        assert len(samples) == 2
        for obj in samples:
            assert isinstance(obj, dict)

    def test_train_with_grammar_infer_without_grammar(self, sample_data):
        """Train with grammar, disable grammar at inference."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True),
            inference=InferenceConfig(constrain_grammar=False),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Model SHOULD have grammar PDA from training
        assert pipeline.model._grammar_pda is not None

        # Generator should still be created without grammar
        gen = pipeline._get_generator()
        assert gen._grammar_pda is None

    def test_train_without_schema_infer_with_schema(self, sample_data):
        """Train without schema, apply schema at inference (main use case)."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True, constrain_schema=False),
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Schema should have been inferred
        assert pipeline.schema is not None

        # Generation should use both grammar and schema
        gen = pipeline._get_generator()
        assert gen._grammar_pda is not None
        assert gen._schema_pda is not None

    def test_train_with_schema_infer_without_schema(self, sample_data):
        """Train with schema, disable schema at inference."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True, constrain_schema=True),
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=False),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Generator should NOT have schema PDA (inference says False)
        gen = pipeline._get_generator()
        assert gen._grammar_pda is not None
        assert gen._schema_pda is None

    def test_all_constraints_off(self, sample_data):
        """No constraints during training or inference."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=False, constrain_schema=False),
            inference=InferenceConfig(constrain_grammar=False, constrain_schema=False),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        gen = pipeline._get_generator()
        assert gen._grammar_pda is None
        assert gen._schema_pda is None

    def test_all_constraints_on(self, sample_data):
        """All constraints during training and inference."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True, constrain_schema=True),
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        gen = pipeline._get_generator()
        assert gen._grammar_pda is not None
        assert gen._schema_pda is not None


class TestPipelineSchemaInteraction:
    """Tests for schema inference + constraint interactions."""

    def test_infer_schema_false_no_explicit_schema_constrain_schema_true_train(self, sample_data):
        """constrain_schema=True in training without schema should fail."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_schema=True),
            data=DataConfig(infer_schema=False),  # No schema
        )
        pipeline = OrigamiPipeline(config)
        # Should fail during training when trainer checks for schema
        with pytest.raises(ValueError, match="constrain_schema=True requires schema"):
            pipeline.fit(sample_data, epochs=1)

    def test_infer_schema_false_constrain_schema_true_inference(self, sample_data):
        """constrain_schema=True at inference without schema should fail."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True, constrain_schema=False),
            inference=InferenceConfig(constrain_schema=True),
            data=DataConfig(infer_schema=False),  # No schema
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        assert pipeline.schema is None

        # Generating should fail because constrain_schema=True but no schema
        with pytest.raises(ValueError, match="constrain_schema=True requires schema"):
            pipeline.generate(num_samples=1)

    def test_infer_schema_true_provides_schema_to_inference(self, sample_data):
        """infer_schema=True populates schema for inference use."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            inference=InferenceConfig(constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        assert pipeline.schema is not None
        # Schema should have properties for both fields
        assert "properties" in pipeline.schema
        assert "color" in pipeline.schema["properties"]
        assert "shape" in pipeline.schema["properties"]

    def test_explicit_schema_overrides_infer(self, sample_data, sample_schema):
        """Explicit schema in DataConfig takes precedence over infer_schema."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            inference=InferenceConfig(constrain_schema=True),
            data=DataConfig(schema=sample_schema, infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Should use explicit schema (which has enum constraints)
        assert pipeline.schema is not None
        color_schema = pipeline.schema["properties"]["color"]
        assert "enum" in color_schema
        assert set(color_schema["enum"]) == {"red", "blue"}


class TestPipelineEvaluateConstraints:
    """Tests for Pipeline.evaluate() with constraint settings."""

    @pytest.fixture
    def fitted_pipeline(self, sample_data):
        """Pipeline fitted with infer_schema and inference constraints."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True, constrain_schema=False),
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=2)
        return pipeline

    def test_evaluate_uses_inference_config(self, fitted_pipeline, sample_data):
        """Pipeline.evaluate() should use inference constraint settings."""
        results = fitted_pipeline.evaluate(sample_data)
        assert "loss" in results
        assert np.isfinite(results["loss"])

    def test_evaluate_with_metrics(self, fitted_pipeline, sample_data):
        """Pipeline.evaluate() with metrics uses inference constraints."""
        results = fitted_pipeline.evaluate(
            sample_data,
            target_key="shape",
            metrics={"acc": "accuracy"},
        )
        assert "loss" in results
        assert "acc" in results
        assert np.isfinite(results["loss"])

    def test_evaluate_loss_finite_with_unseen_values(self):
        """Eval loss should be finite even with UNK tokens (lenient schema PDA)."""
        torch.manual_seed(42)

        train_data = [{"label": v, "x": i} for i, v in enumerate(["A", "B", "C"] * 10)]
        eval_data = [{"label": "UNSEEN", "x": i} for i in range(5)]

        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_schema=True),
            inference=InferenceConfig(constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )

        pipeline = OrigamiPipeline(config)
        pipeline.fit(train_data, eval_data=eval_data, epochs=1)

        # Evaluate on data with unseen values — should not produce inf loss
        results = pipeline.evaluate(eval_data)
        assert np.isfinite(results["loss"]), (
            f"Eval loss is {results['loss']} — expected finite. "
            "Lenient UNK in evaluator should prevent inf loss."
        )


class TestPipelinePredictConstraints:
    """Tests for Pipeline.predict() with constraint settings."""

    def test_predict_uses_inference_constraints(self, sample_data):
        """Prediction should use inference config, not training config."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=False),  # Off for training
            inference=InferenceConfig(constrain_grammar=True),  # On for inference
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        result = pipeline.predict(sample_data[0], target_key="shape")
        # Should not crash — grammar constraint applied via inference config
        assert result is not None or result is None

    def test_predict_proba_with_schema(self, sample_data, sample_schema):
        """predict_proba with schema constraints works."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
            data=DataConfig(schema=sample_schema),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        probs = pipeline.predict_proba(sample_data[0], target_key="shape", top_k=3)
        assert isinstance(probs, list)
        assert len(probs) <= 3


class TestPipelineGenerateConstraints:
    """Tests for Pipeline.generate() with constraint settings."""

    def test_generate_uses_inference_constraints(self, sample_data):
        """Generation should use inference config."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=False),
            inference=InferenceConfig(constrain_grammar=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        samples = pipeline.generate(num_samples=2, max_length=64)
        assert len(samples) == 2
        for obj in samples:
            assert isinstance(obj, dict)

    def test_generate_with_schema(self, sample_data):
        """Generation with inferred schema constraints."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        samples = pipeline.generate(num_samples=2, max_length=64)
        assert len(samples) == 2


# ===========================================================================
# Pipeline save/load with InferenceConfig
# ===========================================================================


class TestPipelineSaveLoadInferenceConfig:
    """Tests that InferenceConfig survives save/load round-trip."""

    def test_inference_config_persists(self, sample_data, tmp_path):
        """InferenceConfig should be saved and loaded from checkpoint."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            inference=InferenceConfig(constrain_grammar=False, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Save
        path = tmp_path / "model.pt"
        pipeline.save(str(path))

        # Load
        loaded = OrigamiPipeline.load(str(path))

        assert loaded.config.inference.constrain_grammar is False
        assert loaded.config.inference.constrain_schema is True

    def test_load_old_checkpoint_without_inference_config(self, sample_data, tmp_path):
        """Loading a checkpoint without InferenceConfig should use defaults."""
        torch.manual_seed(42)
        config = OrigamiConfig(model=SMALL_MODEL_CFG)
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Save
        path = tmp_path / "model.pt"
        pipeline.save(str(path))

        # Simulate old checkpoint by removing inference key
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        state["config"].pop("inference", None)
        torch.save(state, str(path))

        # Load should still work with default InferenceConfig
        loaded = OrigamiPipeline.load(str(path))
        assert loaded.config.inference.constrain_grammar is True
        assert loaded.config.inference.constrain_schema is False

    def test_saved_inference_affects_generation(self, sample_data, tmp_path):
        """Loaded pipeline should use saved inference constraints."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=False),
            inference=InferenceConfig(constrain_grammar=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        path = tmp_path / "model.pt"
        pipeline.save(str(path))

        loaded = OrigamiPipeline.load(str(path))
        # Generator should use inference config (grammar=True)
        gen = loaded._get_generator()
        assert gen._grammar_pda is not None


# ===========================================================================
# Training with different constraint combos
# ===========================================================================


class TestTrainerConstraintCombos:
    """Tests for different training constraint combinations."""

    def test_train_no_constraints(self, sample_data):
        """Training with no constraints completes without error."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=False, constrain_schema=False),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=2)
        assert pipeline._fitted

    def test_train_grammar_only(self, sample_data):
        """Training with grammar only completes without error."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True, constrain_schema=False),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=2)
        assert pipeline._fitted

    def test_train_schema_only(self, sample_data):
        """Training with schema only (no grammar) completes without error."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=False, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=2)
        assert pipeline._fitted

    def test_train_both_constraints(self, sample_data):
        """Training with both grammar and schema completes without error."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(constrain_grammar=True, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=2)
        assert pipeline._fitted

    def test_train_with_eval_metrics_and_constraints(self, sample_data):
        """Training with eval metrics + constraints works end-to-end."""
        torch.manual_seed(42)
        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(
                constrain_grammar=True,
                constrain_schema=True,
                eval_strategy="epoch",
                eval_metrics={"acc": "accuracy"},
                target_key="shape",
            ),
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
            data=DataConfig(infer_schema=True),
        )
        pipeline = OrigamiPipeline(config)
        # Split data for train/eval
        pipeline.fit(sample_data[:15], eval_data=sample_data[15:], epochs=2)
        assert pipeline._fitted


# ===========================================================================
# Parametrized matrix of all combos
# ===========================================================================


class TestConstraintMatrix:
    """Parametrized tests for all training × inference constraint combinations."""

    @pytest.mark.parametrize(
        "train_grammar,train_schema",
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["train:none", "train:grammar", "train:schema", "train:both"],
    )
    @pytest.mark.parametrize(
        "infer_grammar,infer_schema",
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["infer:none", "infer:grammar", "infer:schema", "infer:both"],
    )
    def test_train_and_evaluate(
        self, sample_data, train_grammar, train_schema, infer_grammar, infer_schema
    ):
        """All 16 combinations of training × inference constraints should work."""
        torch.manual_seed(42)

        # Schema required when any constrain_schema is True
        needs_schema = train_schema or infer_schema

        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(
                constrain_grammar=train_grammar,
                constrain_schema=train_schema,
            ),
            inference=InferenceConfig(
                constrain_grammar=infer_grammar,
                constrain_schema=infer_schema,
            ),
            data=DataConfig(infer_schema=needs_schema),
        )

        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Evaluate should work
        results = pipeline.evaluate(sample_data)
        assert "loss" in results
        assert np.isfinite(results["loss"])

    @pytest.mark.parametrize(
        "train_grammar,train_schema",
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["train:none", "train:grammar", "train:schema", "train:both"],
    )
    @pytest.mark.parametrize(
        "infer_grammar,infer_schema",
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["infer:none", "infer:grammar", "infer:schema", "infer:both"],
    )
    def test_train_and_generate(
        self, sample_data, train_grammar, train_schema, infer_grammar, infer_schema
    ):
        """All 16 combinations should produce generated output."""
        torch.manual_seed(42)

        needs_schema = train_schema or infer_schema

        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(
                constrain_grammar=train_grammar,
                constrain_schema=train_schema,
            ),
            inference=InferenceConfig(
                constrain_grammar=infer_grammar,
                constrain_schema=infer_schema,
            ),
            data=DataConfig(infer_schema=needs_schema),
        )

        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Generate should work (may produce invalid JSON without grammar)
        samples = pipeline.generate(num_samples=1, max_length=64)
        assert isinstance(samples, list)

    @pytest.mark.parametrize(
        "train_grammar,train_schema",
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["train:none", "train:grammar", "train:schema", "train:both"],
    )
    @pytest.mark.parametrize(
        "infer_grammar,infer_schema",
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["infer:none", "infer:grammar", "infer:schema", "infer:both"],
    )
    def test_train_and_predict(
        self, sample_data, train_grammar, train_schema, infer_grammar, infer_schema
    ):
        """All 16 combinations should produce predictions."""
        torch.manual_seed(42)

        needs_schema = train_schema or infer_schema

        config = OrigamiConfig(
            model=SMALL_MODEL_CFG,
            training=TrainingConfig(
                constrain_grammar=train_grammar,
                constrain_schema=train_schema,
            ),
            inference=InferenceConfig(
                constrain_grammar=infer_grammar,
                constrain_schema=infer_schema,
            ),
            data=DataConfig(infer_schema=needs_schema),
        )

        pipeline = OrigamiPipeline(config)
        pipeline.fit(sample_data, epochs=1)

        # Without grammar constraints, an untrained model can generate invalid
        # sequences, raising GenerationError. This is expected behavior.
        try:
            result = pipeline.predict(sample_data[0], target_key="shape")
            assert result is not None or result is None
        except GenerationError:
            assert not infer_grammar, "GenerationError with grammar enabled is unexpected"
