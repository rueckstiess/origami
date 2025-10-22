import pytest
from mdbrtools.query import Predicate, Query
from unittest.mock import Mock

from origami.inference import RejectionEstimator


@pytest.fixture
def mock_sampler():
    """Create a mock sampler for testing."""
    sampler = Mock()
    return sampler


@pytest.fixture
def rejection_estimator(mock_sampler):
    """Create a RejectionEstimator instance."""
    return RejectionEstimator(mock_sampler)


class TestRejectionEstimatorConstructor:
    """Test RejectionEstimator constructor."""

    def test_constructor(self, mock_sampler):
        """Test that constructor properly initializes."""
        estimator = RejectionEstimator(mock_sampler)

        assert estimator.sampler is mock_sampler


class TestMatchesQuery:
    """Test _matches_query method."""

    def test_matches_query_single_predicate_match(self, rejection_estimator):
        """Test matching a document with single predicate."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (5.0,)))

        doc = {"a": 7.0, "b": 2.0}
        assert rejection_estimator._matches_query(doc, query) is True

    def test_matches_query_single_predicate_no_match(self, rejection_estimator):
        """Test non-matching document with single predicate."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (5.0,)))

        doc = {"a": 3.0, "b": 2.0}
        assert rejection_estimator._matches_query(doc, query) is False

    def test_matches_query_range_match(self, rejection_estimator):
        """Test matching with range predicates."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        doc = {"a": 5.0, "b": 2.0}
        assert rejection_estimator._matches_query(doc, query) is True

    def test_matches_query_range_no_match(self, rejection_estimator):
        """Test non-matching with range predicates."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("a", "lte", (7.0,)))

        doc = {"a": 10.0, "b": 2.0}
        assert rejection_estimator._matches_query(doc, query) is False

    def test_matches_query_multiple_fields(self, rejection_estimator):
        """Test matching with predicates on multiple fields."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("b", "lte", (5.0,)))

        doc = {"a": 5.0, "b": 3.0}
        assert rejection_estimator._matches_query(doc, query) is True

    def test_matches_query_multiple_fields_partial_match(self, rejection_estimator):
        """Test that partial match fails (AND logic)."""
        query = Query()
        query.add_predicate(Predicate("a", "gte", (3.0,)))
        query.add_predicate(Predicate("b", "lte", (5.0,)))

        # Matches first predicate but not second
        doc = {"a": 5.0, "b": 10.0}
        assert rejection_estimator._matches_query(doc, query) is False


class TestEstimateMethod:
    """Test estimate method logic."""

    def test_estimate_all_accepted(self, mock_sampler):
        """Test when all samples match the query."""
        # Mock sampler to return documents that all match
        mock_sampler.sample.return_value = (
            [{"a": 5.0}, {"a": 6.0}, {"a": 7.0}],  # documents
            [0.1, 0.1, 0.1]  # log_probs (not used in rejection estimator)
        )

        estimator = RejectionEstimator(mock_sampler)

        query = Query()
        query.add_predicate(Predicate("a", "gte", (4.0,)))

        selectivity, accepted = estimator.estimate(query, n=3)

        # All 3 documents match
        assert selectivity == 1.0
        assert len(accepted) == 3

    def test_estimate_none_accepted(self, mock_sampler):
        """Test when no samples match the query."""
        # Mock sampler to return documents that don't match
        mock_sampler.sample.return_value = (
            [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}],
            [0.1, 0.1, 0.1]
        )

        estimator = RejectionEstimator(mock_sampler)

        query = Query()
        query.add_predicate(Predicate("a", "gte", (10.0,)))

        selectivity, accepted = estimator.estimate(query, n=3)

        # No documents match
        assert selectivity == 0.0
        assert len(accepted) == 0

    def test_estimate_partial_acceptance(self, mock_sampler):
        """Test when some samples match."""
        # Mock sampler to return mixed documents
        mock_sampler.sample.return_value = (
            [{"a": 5.0}, {"a": 15.0}, {"a": 7.0}, {"a": 20.0}],
            [0.1, 0.1, 0.1, 0.1]
        )

        estimator = RejectionEstimator(mock_sampler)

        query = Query()
        query.add_predicate(Predicate("a", "lte", (10.0,)))

        selectivity, accepted = estimator.estimate(query, n=4)

        # 2 out of 4 match (5.0 and 7.0)
        assert selectivity == 0.5
        assert len(accepted) == 2
        assert {"a": 5.0} in accepted
        assert {"a": 7.0} in accepted

    def test_estimate_return_samples_false(self, mock_sampler):
        """Test that return_samples=False returns None for samples."""
        mock_sampler.sample.return_value = (
            [{"a": 5.0}, {"a": 6.0}],
            [0.1, 0.1]
        )

        estimator = RejectionEstimator(mock_sampler)

        query = Query()
        query.add_predicate(Predicate("a", "gte", (4.0,)))

        selectivity, accepted = estimator.estimate(query, n=2, return_samples=False)

        assert selectivity == 1.0
        assert accepted is None

    def test_estimate_calls_sampler_with_correct_n(self, mock_sampler):
        """Test that estimator requests correct number of samples."""
        mock_sampler.sample.return_value = ([{"a": 5.0}] * 100, [0.1] * 100)

        estimator = RejectionEstimator(mock_sampler)

        query = Query()
        query.add_predicate(Predicate("a", "gte", (4.0,)))

        estimator.estimate(query, n=100)

        # Verify sampler.sample was called with n=100
        mock_sampler.sample.assert_called_once_with(n=100)


class TestEstimateIntegration:
    """Integration tests for estimate behavior."""

    def test_selectivity_in_valid_range(self, mock_sampler):
        """Test that selectivity is always between 0 and 1."""
        mock_sampler.sample.return_value = (
            [{"a": i} for i in range(10)],
            [0.1] * 10
        )

        estimator = RejectionEstimator(mock_sampler)

        query = Query()
        query.add_predicate(Predicate("a", "gte", (5.0,)))

        selectivity, _ = estimator.estimate(query, n=10)

        assert 0.0 <= selectivity <= 1.0

    def test_different_query_sizes(self, mock_sampler):
        """Test with different sample sizes."""
        for n in [10, 50, 100]:
            mock_sampler.sample.return_value = (
                [{"a": 5.0}] * n,
                [0.1] * n
            )

            estimator = RejectionEstimator(mock_sampler)

            query = Query()
            query.add_predicate(Predicate("a", "gte", (4.0,)))

            selectivity, accepted = estimator.estimate(query, n=n)

            assert selectivity == 1.0
            assert len(accepted) == n
