from typing import Optional

from mdbrtools.query import Query

from origami.inference.sampler import Sampler
from origami.utils.query_utils import _doc_matches_query


class RejectionEstimator:
    """Estimate query selectivity using rejection sampling.

    Samples from the learned model distribution and counts what fraction
    of samples satisfy the query predicates. This gives an unbiased estimate
    of the query selectivity.

    The estimator works by:
    1. Sampling documents from the learned model distribution
    2. Rejecting samples that don't match the query predicates
    3. Estimating selectivity as: acceptance_rate = (accepted / total)

    This is an unbiased estimator because the acceptance rate equals P(query).

    Args:
        sampler: Sampler instance for generating documents from model distribution

    Example:
        >>> from origami.inference import Sampler, RejectionEstimator
        >>> sampler = Sampler(model, encoder, schema)
        >>> estimator = RejectionEstimator(sampler)
        >>>
        >>> query = Query()
        >>> query.add_predicate(Predicate('a', 'gte', (3.0,)))
        >>> query.add_predicate(Predicate('a', 'lte', (7.0,)))
        >>>
        >>> selectivity, accepted = estimator.estimate(query, n=1000)
        >>> cardinality = selectivity * collection_size
        >>> print(f"Estimated {len(accepted)}/1000 samples match the query")
    """

    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def estimate(self, query: Query, n: int = 1000, return_samples: bool = True) -> tuple[float, Optional[list[dict]]]:
        """Estimate query selectivity using rejection sampling.

        Args:
            query: MongoDB query object (from mdbrtools.query)
            n: Number of samples to generate
            return_samples: If True, return accepted samples; if False, return None

        Returns:
            Tuple of (selectivity, accepted_samples):
                - selectivity: Estimated P(query) in range [0, 1]
                  Multiply by collection_size to get cardinality estimate
                - accepted_samples: List of samples that matched the query,
                  or None if return_samples=False

        Example:
            >>> selectivity, accepted = estimator.estimate(query, n=1000)
            >>> print(f"Selectivity: {selectivity:.4f}")
            >>> print(f"Accepted samples: {len(accepted)}")
            >>>
            >>> # For large n, save memory by not returning samples
            >>> selectivity, _ = estimator.estimate(query, n=100000, return_samples=False)
        """
        # 1. Sample from model distribution
        documents, _ = self.sampler.sample(n=n)

        # 2. Filter by query (rejection step)
        accepted = []
        for doc in documents:
            if self._matches_query(doc, query):
                accepted.append(doc)

        # 3. Unbiased estimate: acceptance rate = P(query)
        selectivity = len(accepted) / n

        # Return samples only if requested (saves memory for large n)
        return selectivity, accepted if return_samples else None

    def _matches_query(self, doc: dict, query: Query) -> bool:
        """Check if a document matches all query predicates.

        Uses AND logic: document must satisfy all predicates.

        Args:
            doc: Document to check
            query: Query with predicates to evaluate

        Returns:
            True if document matches all predicates, False otherwise
        """
        return _doc_matches_query(doc, query)
