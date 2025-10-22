import math

import numpy as np
import pandas as pd
import pytest
import torch
from mdbrtools.query import Predicate, Query
from sklearn.pipeline import Pipeline

from origami.inference import MCEstimator
from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import (
    DFDataset,
    DocTokenizerPipe,
    KBinsDiscretizerPipe,
    PadTruncTokensPipe,
    SchemaParserPipe,
    SortFieldsPipe,
    TokenEncoderPipe,
)
from origami.utils import ModelConfig, TrainConfig


@pytest.fixture
def simple_2d_dataset():
    """Create a simple 2D Gaussian dataset for testing."""
    np.random.seed(42)
    n_docs = 1000

    # Create simple 2D data
    a_values = np.random.uniform(0, 10, n_docs)
    b_values = np.random.uniform(-5, 5, n_docs)

    docs = [{"a": float(a), "b": float(b)} for a, b in zip(a_values, b_values)]
    df = pd.DataFrame({"docs": docs})

    return df, docs


@pytest.fixture
def pipeline_and_model(simple_2d_dataset):
    """Create a fitted pipeline and trained model."""
    df, _ = simple_2d_dataset

    # Create pipeline
    pipeline = Pipeline(
        [
            ("binning", KBinsDiscretizerPipe(bins=10, strategy="uniform")),
            ("schema", SchemaParserPipe()),
            ("tokenizer", DocTokenizerPipe()),
            ("padding", PadTruncTokensPipe(length="max")),
            ("encoder", TokenEncoderPipe()),
        ]
    )

    # Fit and transform
    processed_df = pipeline.fit_transform(df).reset_index(drop=True)

    # Get components
    schema = pipeline["schema"].schema
    encoder = pipeline["encoder"].encoder
    block_size = pipeline["padding"].length

    # Create model
    model_config = ModelConfig.from_preset("xs")
    model_config.vocab_size = encoder.vocab_size
    model_config.block_size = block_size
    model_config.position_encoding = "KEY_VALUE"

    train_config = TrainConfig()
    train_config.learning_rate = 1e-3

    vpda = ObjectVPDA(encoder, schema)
    model = ORIGAMI(model_config, train_config, vpda=vpda)

    # Quick training (just a few batches to get reasonable probabilities)
    dataset = DFDataset(processed_df)
    model.train_model(dataset, batches=100)

    return pipeline, model


@pytest.fixture
def mc_estimator(pipeline_and_model):
    """Create an MCEstimator instance."""
    pipeline, model = pipeline_and_model
    return MCEstimator(model, pipeline, batch_size=100)


class TestMCEstimatorConstructor:
    """Test MCEstimator constructor and initialization."""

    def test_constructor_extracts_components(self, pipeline_and_model):
        """Test that constructor properly extracts components from pipeline."""
        pipeline, model = pipeline_and_model
        estimator = MCEstimator(model, pipeline)

        assert estimator.model is model
        assert estimator.pipeline is pipeline
        assert estimator.encoder is pipeline["encoder"].encoder
        assert estimator.schema is pipeline["schema"].schema
        assert estimator.discretizers is pipeline["binning"].discretizers
        assert estimator.batch_size == 1000

    def test_constructor_custom_batch_size(self, pipeline_and_model):
        """Test constructor with custom batch size."""
        pipeline, model = pipeline_and_model
        estimator = MCEstimator(model, pipeline, batch_size=500)

        assert estimator.batch_size == 500

    def test_constructor_detects_no_permuter(self, pipeline_and_model):
        """Test that constructor correctly detects absence of permuter."""
        pipeline, model = pipeline_and_model
        estimator = MCEstimator(model, pipeline)

        assert estimator.has_permuter is False

    def test_constructor_detects_no_sorter(self, pipeline_and_model):
        """Test that constructor correctly detects absence of sorter."""
        pipeline, model = pipeline_and_model
        estimator = MCEstimator(model, pipeline)

        assert estimator.has_sorter is False


class TestGetAllowedValues:
    """Test _get_allowed_values method."""

    def test_single_gte_predicate(self, mc_estimator):
        """Test filtering with single $gte predicate."""
        predicate = Predicate("a", "gte", (5.0,))
        allowed = mc_estimator._get_allowed_values("a", [predicate])

        # All values should be >= 5.0
        assert all(v >= 5.0 for v in allowed)
        assert len(allowed) > 0

    def test_single_lte_predicate(self, mc_estimator):
        """Test filtering with single $lte predicate."""
        predicate = Predicate("a", "lte", (5.0,))
        allowed = mc_estimator._get_allowed_values("a", [predicate])

        # All values should be <= 5.0
        assert all(v <= 5.0 for v in allowed)
        assert len(allowed) > 0

    def test_range_predicates(self, mc_estimator):
        """Test filtering with range predicates ($gte and $lte)."""
        pred_gte = Predicate("a", "gte", (3.0,))
        pred_lte = Predicate("a", "lte", (7.0,))
        allowed = mc_estimator._get_allowed_values("a", [pred_gte, pred_lte])

        # All values should be in [3.0, 7.0]
        assert all(3.0 <= v <= 7.0 for v in allowed)
        assert len(allowed) > 0

    def test_eq_predicate(self, mc_estimator):
        """Test filtering with $eq predicate."""
        # Get a value from the schema
        all_values = list(mc_estimator.schema.get_prim_values("a"))
        target_value = all_values[len(all_values) // 2]

        predicate = Predicate("a", "eq", (target_value,))
        allowed = mc_estimator._get_allowed_values("a", [predicate])

        # Should return only the matching value
        assert len(allowed) == 1
        assert allowed[0] == target_value

    def test_empty_result(self, mc_estimator):
        """Test that contradictory predicates return empty list."""
        pred_gt = Predicate("a", "gt", (100.0,))  # No values above 10
        allowed = mc_estimator._get_allowed_values("a", [pred_gt])

        assert len(allowed) == 0


class TestGenerateUniformSamples:
    """Test _generate_uniform_samples method."""

    def test_generates_correct_number(self, mc_estimator):
        """Test that correct number of samples is generated."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        samples = mc_estimator._generate_uniform_samples(query, n=50)

        assert len(samples) == 50

    def test_all_schema_fields_present(self, mc_estimator):
        """Test that all schema fields are present in generated documents."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))

        samples = mc_estimator._generate_uniform_samples(query, n=10)

        schema_fields = set(mc_estimator.schema.leaf_fields.keys())
        for doc in samples:
            assert set(doc.keys()) == schema_fields

    def test_query_fields_have_allowed_values(self, mc_estimator):
        """Test that query fields only have values within query constraints."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        samples = mc_estimator._generate_uniform_samples(query, n=100)

        # All 'a' values should be in [3.0, 7.0]
        for doc in samples:
            assert 3.0 <= doc["a"] <= 7.0

    def test_non_query_fields_have_any_value(self, mc_estimator):
        """Test that non-query fields can have any schema value."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        samples = mc_estimator._generate_uniform_samples(query, n=100)

        # Field 'b' is not constrained, should have full range
        b_values = [doc["b"] for doc in samples]
        b_range = max(b_values) - min(b_values)

        # Should span a reasonable portion of the full range
        # (not just a tiny interval)
        assert b_range > 2.0  # Original range is [-5, 5]

    def test_samples_with_sorted_fields(self, simple_2d_dataset):
        """Test that samples are sorted when SortFieldsPipe is in pipeline."""
        df, _ = simple_2d_dataset

        # Create pipeline WITH SortFieldsPipe
        pipeline = Pipeline(
            [
                ("binning", KBinsDiscretizerPipe(bins=10, strategy="uniform")),
                ("sorter", SortFieldsPipe()),  # Add sorter
                ("schema", SchemaParserPipe()),
                ("tokenizer", DocTokenizerPipe()),
                ("padding", PadTruncTokensPipe(length="max")),
                ("encoder", TokenEncoderPipe()),
            ]
        )

        processed_df = pipeline.fit_transform(df).reset_index(drop=True)

        schema = pipeline["schema"].schema
        encoder = pipeline["encoder"].encoder
        block_size = pipeline["padding"].length

        model_config = ModelConfig.from_preset("xs")
        model_config.vocab_size = encoder.vocab_size
        model_config.block_size = block_size
        model_config.position_encoding = "KEY_VALUE"

        train_config = TrainConfig()
        train_config.learning_rate = 1e-3

        vpda = ObjectVPDA(encoder, schema)
        model = ORIGAMI(model_config, train_config, vpda=vpda)

        dataset = DFDataset(processed_df)
        model.train_model(dataset, batches=10)

        # Create estimator
        estimator = MCEstimator(model, pipeline, batch_size=100)

        # Verify has_sorter is True
        assert estimator.has_sorter is True

        # Generate samples
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        samples = estimator._generate_uniform_samples(query, n=10)

        # All samples should have alphabetically sorted keys
        from collections import OrderedDict

        for doc in samples:
            assert isinstance(doc, OrderedDict)
            keys = list(doc.keys())
            assert keys == sorted(keys), f"Keys {keys} are not sorted"


class TestCalculateQueryRegionSize:
    """Test _calculate_query_region_size method."""

    def test_query_single_field(self, mc_estimator):
        """Test |E| calculation for query on single field."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        E_size = mc_estimator._calculate_query_region_size(query)

        # E_size = (filtered count for a) * (all values for b)
        predicates_a = [p for p in query.predicates if p.column == "a"]
        allowed_a = mc_estimator._get_allowed_values("a", predicates_a)
        all_b = list(mc_estimator.schema.get_prim_values("b"))

        expected = len(allowed_a) * len(all_b)
        assert E_size == expected

    def test_query_multiple_fields(self, mc_estimator):
        """Test |E| calculation for query on multiple fields."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))
        query.add_predicate(Predicate("b", "gte", (-2.0,)))
        query.add_predicate(Predicate("b", "lte", (2.0,)))

        E_size = mc_estimator._calculate_query_region_size(query)

        # E_size = (filtered count for a) * (filtered count for b)
        predicates_a = [p for p in query.predicates if p.column == "a"]
        allowed_a = mc_estimator._get_allowed_values("a", predicates_a)

        predicates_b = [p for p in query.predicates if p.column == "b"]
        allowed_b = mc_estimator._get_allowed_values("b", predicates_b)

        expected = len(allowed_a) * len(allowed_b)
        assert E_size == expected

    def test_empty_query_region(self, mc_estimator):
        """Test that empty query region returns 0."""
        query = Query()
        query.add_predicate(Predicate("a", "gt", (100.0,)))  # No values > 10

        E_size = mc_estimator._calculate_query_region_size(query)

        assert E_size == 0


class TestComputeModelProbabilities:
    """Test _compute_model_probabilities method."""

    def test_returns_correct_shape(self, mc_estimator, simple_2d_dataset):
        """Test that probabilities array has correct shape."""
        _, docs = simple_2d_dataset
        test_docs = docs[:10]

        probs = mc_estimator._compute_model_probabilities(test_docs)

        assert isinstance(probs, np.ndarray)
        assert len(probs) == 10

    def test_probabilities_are_positive(self, mc_estimator, simple_2d_dataset):
        """Test that all probabilities are positive."""
        _, docs = simple_2d_dataset
        test_docs = docs[:20]

        probs = mc_estimator._compute_model_probabilities(test_docs)

        assert np.all(probs > 0)

    def test_probabilities_reasonable_magnitude(self, mc_estimator, simple_2d_dataset):
        """Test that probabilities are in reasonable range."""
        _, docs = simple_2d_dataset
        test_docs = docs[:20]

        probs = mc_estimator._compute_model_probabilities(test_docs)

        # Probabilities should be between 0 and 1 (before E_size multiplication)
        # But these are raw model probs, can be small
        assert np.all(probs < 1.0)
        assert np.all(probs > 1e-10)


class TestEstimateIntegration:
    """Integration tests for estimate method."""

    def test_estimate_returns_probability(self, mc_estimator):
        """Test that estimate returns a probability value and samples."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        prob, samples = mc_estimator.estimate(query, n=100)

        assert isinstance(prob, float)
        assert prob >= 0.0
        assert isinstance(samples, list)
        assert len(samples) == 100

    def test_estimate_empty_query_returns_zero(self, mc_estimator):
        """Test that empty query region returns 0.0 and empty samples."""
        query = Query()
        query.add_predicate(Predicate("a", "gt", (100.0,)))  # No values > 10

        prob, samples = mc_estimator.estimate(query, n=100)

        assert prob == 0.0
        assert samples == []

    def test_estimate_full_range_gives_high_probability(self, mc_estimator):
        """Test that full range query gives higher probability than narrow query."""
        # Full range query
        query_full = Query()
        query_full.add_predicate(Predicate("a", "gte", (0.0,)))
        query_full.add_predicate(Predicate("a", "lte", (10.0,)))
        query_full.add_predicate(Predicate("b", "gte", (-5.0,)))
        query_full.add_predicate(Predicate("b", "lte", (5.0,)))

        # Narrow query (25% of range)
        query_narrow = Query()
        query_narrow.add_predicate(Predicate("a", "gte", (4.0,)))
        query_narrow.add_predicate(Predicate("a", "lte", (6.0,)))
        query_narrow.add_predicate(Predicate("b", "gte", (-1.0,)))
        query_narrow.add_predicate(Predicate("b", "lte", (1.0,)))

        prob_full, _ = mc_estimator.estimate(query_full, n=100)
        prob_narrow, _ = mc_estimator.estimate(query_narrow, n=100)

        # Full range should have significantly higher probability
        assert prob_full > prob_narrow
        assert prob_full > 0.2  # At least some reasonable probability

    def test_estimate_consistency(self, mc_estimator):
        """Test that repeated estimates are reasonably consistent."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        # Run multiple estimates
        estimates = [mc_estimator.estimate(query, n=200)[0] for _ in range(5)]

        # Check consistency (coefficient of variation < 0.3)
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        cv = std_est / mean_est if mean_est > 0 else 0

        assert cv < 0.3  # Reasonable variance for MC estimation

    def test_smaller_region_gives_smaller_probability(self, mc_estimator):
        """Test that smaller query regions give smaller probabilities."""
        # Large region
        query_large = Query()
        query_large.add_predicate(Predicate("a", "gte", (2.0,)))
        query_large.add_predicate(Predicate("a", "lte", (8.0,)))

        # Small region
        query_small = Query()
        query_small.add_predicate(Predicate("a", "gte", (4.5,)))
        query_small.add_predicate(Predicate("a", "lte", (5.5,)))

        prob_large, _ = mc_estimator.estimate(query_large, n=200)
        prob_small, _ = mc_estimator.estimate(query_small, n=200)

        # Larger region should have higher probability
        assert prob_large > prob_small

    def test_samples_match_query_constraints(self, mc_estimator):
        """Test that returned samples satisfy query constraints."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))
        query.add_predicate(Predicate("b", "gte", (-2.0,)))
        query.add_predicate(Predicate("b", "lte", (2.0,)))

        prob, samples = mc_estimator.estimate(query, n=50)

        # All samples should satisfy the query constraints
        for doc in samples:
            assert 3.0 <= doc["a"] <= 7.0
            assert -2.0 <= doc["b"] <= 2.0

    def test_samples_are_complete_documents(self, mc_estimator):
        """Test that samples contain all schema fields."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))

        prob, samples = mc_estimator.estimate(query, n=20)

        schema_fields = set(mc_estimator.schema.leaf_fields.keys())
        for doc in samples:
            assert set(doc.keys()) == schema_fields
