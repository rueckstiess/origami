"""Utility functions for working with queries."""

from typing import Any, Dict, List

from mdbrtools.query import Query

from origami.utils.common import OPERATORS, mult_error


def evaluate_ground_truth(query: Query, docs: List[Dict[str, Any]]) -> int:
    """Calculate ground truth cardinality by checking which documents match the query.

    Args:
        query: MongoDB query object (from mdbrtools.query)
        docs: List of documents to evaluate

    Returns:
        Number of documents that match all query predicates

    Example:
        >>> from mdbrtools.query import parse_from_mql
        >>> query = parse_from_mql({"a": {"$gte": 2, "$lte": 5}})
        >>> docs = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 6, "b": 7}]
        >>> evaluate_ground_truth(query, docs)
        1
    """
    count = 0

    for doc in docs:
        if _doc_matches_query(doc, query):
            count += 1

    return count


def _doc_matches_query(doc: Dict[str, Any], query: Query) -> bool:
    """Check if a document matches all predicates in a query.

    Args:
        doc: Document to check
        query: Query with predicates to evaluate

    Returns:
        True if document matches all predicates (AND logic), False otherwise
    """
    for predicate in query.predicates:
        field = predicate.column
        op = OPERATORS[predicate.op]

        # Get field value from document (return None if field doesn't exist)
        value = doc.get(field)

        # Evaluate predicate
        if not op(value, predicate.values):
            return False

    return True


def calculate_selectivity(query: Query, docs: List[Dict[str, Any]]) -> float:
    """Calculate ground truth selectivity (fraction of documents matching query).

    Args:
        query: MongoDB query object
        docs: List of documents to evaluate

    Returns:
        Selectivity in range [0, 1]

    Example:
        >>> from mdbrtools.query import parse_from_mql
        >>> query = parse_from_mql({"a": {"$gte": 2, "$lte": 5}})
        >>> docs = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 6, "b": 7}]
        >>> calculate_selectivity(query, docs)
        0.3333333333333333
    """
    if len(docs) == 0:
        return 0.0

    matching_count = evaluate_ground_truth(query, docs)
    return matching_count / len(docs)


def compare_estimate_to_ground_truth(
    query: Query,
    docs: List[Dict[str, Any]],
    estimated_probability: float,
) -> Dict[str, Any]:
    """Compare estimated selectivity to ground truth and calculate error metrics.

    Args:
        query: MongoDB query object
        docs: List of documents for ground truth
        estimated_probability: Estimated selectivity from MCEstimator

    Returns:
        Dictionary with comparison metrics:
            - ground_truth_count: Actual number of matching documents
            - ground_truth_selectivity: Actual selectivity
            - estimated_count: Estimated count (probability * num_docs)
            - estimated_selectivity: Estimated selectivity
            - absolute_error: |estimated - actual|
            - relative_error: absolute_error / actual (if actual > 0)
            - q_error: max(est, actual) / min(est, actual)

    Example:
        >>> from mdbrtools.query import parse_from_mql
        >>> query = parse_from_mql({"a": {"$gte": 2, "$lte": 5}})
        >>> docs = [{"a": i} for i in range(10)]
        >>> estimated_prob = 0.4  # MCEstimator result
        >>> result = compare_estimate_to_ground_truth(query, docs, estimated_prob)
        >>> print(result["q_error"])
    """

    # Ground truth
    gt_count = evaluate_ground_truth(query, docs)
    gt_selectivity = gt_count / len(docs) if len(docs) > 0 else 0.0

    # Estimated values
    est_count = estimated_probability * len(docs)
    est_selectivity = estimated_probability

    # Error metrics
    abs_error = abs(est_count - gt_count)
    rel_error = abs_error / gt_count if gt_count > 0 else float("inf")
    q_error = mult_error(est_count, gt_count)

    return {
        "ground_truth_count": gt_count,
        "ground_truth_selectivity": gt_selectivity,
        "estimated_count": est_count,
        "estimated_selectivity": est_selectivity,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "q_error": q_error,
    }
