import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mdbrtools.query import Query
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from origami.model.origami import ORIGAMI
from origami.preprocessing import DFDataset
from origami.preprocessing.pipes import DocPermuterPipe, SortFieldsPipe
from origami.utils import Symbol
from origami.utils.common import OPERATORS, sort_dict_fields


class MCEstimator:
    """Monte Carlo cardinality estimator for ORiGAMi models.

    Estimates query selectivity using Monte Carlo sampling by:
    1. Generating uniform samples in the query region
    2. Computing model probabilities f(x) for each sample
    3. Estimating P(query) = |E| * mean(f(x))

    Args:
        model: Trained ORiGAMi model
        pipeline: Fitted preprocessing pipeline (must include encoder, schema, binning)
        batch_size: Batch size for model inference (default: 1000)

    Example:
        >>> estimator = MCEstimator(model, pipeline)
        >>> query = Query()
        >>> query.add_predicate(Predicate('a', 'gte', 2.0))
        >>> query.add_predicate(Predicate('a', 'lte', 5.0))
        >>> prob = estimator.estimate(query, n=1000)
        >>> cardinality = prob * collection_size
    """

    def __init__(
        self,
        model: ORIGAMI,
        pipeline: Pipeline,
        batch_size: int = 1000,
    ):
        self.model = model
        self.pipeline = pipeline
        self.batch_size = batch_size

        # Extract components from pipeline
        self.encoder = pipeline["encoder"].encoder
        self.schema = pipeline["schema"].schema
        self.discretizers = pipeline["binning"].discretizers

        # Detect pipeline components by class type (not name)
        self.has_permuter = any(isinstance(step, DocPermuterPipe) for _, step in pipeline.steps)
        self.has_sorter = any(isinstance(step, SortFieldsPipe) for _, step in pipeline.steps)

    @torch.no_grad()
    def estimate(self, query: Query, n: int = 1000) -> tuple[float, list[dict]]:
        """Estimate the probability (selectivity) of the query region.

        Args:
            query: MongoDB query object (from mdbrtools.query)
            n: Number of uniform samples to generate

        Returns:
            Tuple of (probability, samples):
                - probability: Estimated probability in range [0, 1]
                  Multiply by collection_size to get cardinality estimate
                  Returns 0.0 if query region is empty
                - samples: List of generated uniform sample documents

        Example:
            >>> prob, samples = estimator.estimate(query, n=1000)
            >>> cardinality = prob * collection_size
            >>> print(f"First sample: {samples[0]}")
        """
        # 1. Calculate |E| (query region size)
        E_size = self._calculate_query_region_size(query)

        # Handle empty query region
        if E_size == 0:
            return 0.0, []

        # 2. Generate uniform samples in query region
        docs = self._generate_uniform_samples(query, n)

        # 3. Compute model probabilities f(x)
        probs = self._compute_model_probabilities(docs)

        # 4. Monte Carlo estimate: S = |E| * mean(probs)
        probability = E_size * np.mean(probs)

        return probability, docs

    def _get_allowed_values(self, field: str, predicates: list) -> list:
        """Get allowed values for a field given query predicates.

        Args:
            field: Field name
            predicates: List of Predicate objects for this field

        Returns:
            List of allowed values after applying all predicates
        """
        # Start with all possible values from schema
        allowed_values = list(self.schema.get_prim_values(field))

        # Apply each predicate to narrow the set (AND logic)
        for predicate in predicates:
            op = OPERATORS[predicate.op]
            allowed_values = [x for x in allowed_values if op(x, predicate.values)]

        return allowed_values

    def _generate_uniform_samples(self, query: Query, n: int) -> list[dict]:
        """Generate n uniform samples in the query region.

        Generates complete documents with all schema fields:
        - Query fields: uniformly sample from filtered values
        - Non-query fields: uniformly sample from all values

        Args:
            query: MongoDB query object
            n: Number of samples to generate

        Returns:
            List of n complete documents (dicts)
        """
        # Get all fields from schema
        all_schema_fields = list(self.schema.leaf_fields.keys())
        query_fields = set(query.get_fields())

        # Prepare allowed values for each field
        field_allowed_values = {}

        for field in all_schema_fields:
            if field in query_fields:
                # Query field: filter by predicates
                predicates = [p for p in query.predicates if p.column == field]
                field_allowed_values[field] = self._get_allowed_values(field, predicates)
            else:
                # Non-query field: all values allowed
                field_allowed_values[field] = list(self.schema.get_prim_values(field))

        # Generate n complete documents
        samples = []
        for _ in range(n):
            doc = {field: random.choice(field_allowed_values[field]) for field in all_schema_fields}
            # Apply field ordering to match training pipeline
            if self.has_sorter:
                # Sort fields alphabetically if SortFieldsPipe was used in training
                doc = sort_dict_fields(doc)

            samples.append(doc)

        return samples

    def _calculate_query_region_size(self, query: Query) -> int:
        """Calculate |E| - the number of discrete states in the query region.

        |E| = product of allowed value counts across all schema fields.

        Args:
            query: MongoDB query object

        Returns:
            Number of discrete states in query region
        """
        E_size = 1

        all_schema_fields = list(self.schema.leaf_fields.keys())
        query_fields = set(query.get_fields())

        for field in all_schema_fields:
            if field in query_fields:
                # Query field: count filtered values
                predicates = [p for p in query.predicates if p.column == field]
                allowed_values = self._get_allowed_values(field, predicates)
                E_size *= len(allowed_values)
            else:
                # Non-query field: count all values
                all_values = list(self.schema.get_prim_values(field))
                E_size *= len(all_values)

        return E_size

    def _compute_model_probabilities(self, docs: list[dict]) -> np.ndarray:
        """Compute model probabilities f(x) for a list of documents.

        Args:
            docs: List of complete documents

        Returns:
            Numpy array of probabilities (one per document)
        """
        # 1. Process docs through pipeline
        df = pd.DataFrame({"docs": docs})
        processed_df = self.pipeline.transform(df).reset_index(drop=True)
        dataset = DFDataset(processed_df)

        # 2. Setup batched inference
        self.model.eval()
        loader = DataLoader(dataset, shuffle=False, collate_fn=self.model.collate_fn, batch_size=self.batch_size)

        # 3. Compute log probabilities (for numerical stability)
        log_probs_per_doc = []
        pad_token = self.encoder.encode(Symbol.PAD)

        with torch.no_grad():
            for inputs, targets, masks in loader:
                # Forward pass
                logits, _ = self.model(inputs, targets, masks)

                # Log probabilities
                log_probs = F.log_softmax(logits, dim=-1)

                # Gather target log probs
                target_log_probs = log_probs.gather(dim=2, index=targets.unsqueeze(-1)).squeeze(-1)

                # Mask padding tokens and sum across sequence
                non_pad_mask = (targets != pad_token).float()
                doc_log_probs = (target_log_probs * non_pad_mask).sum(dim=1)

                log_probs_per_doc.extend(doc_log_probs.cpu().tolist())

        # 4. Convert log probs to probs
        probs = np.exp(log_probs_per_doc)

        # 5. Correct for permuter if present
        # Model learned P(doc, ordering) but we want P(doc)
        # Need to marginalize over all orderings
        if self.has_permuter:
            num_fields = len(self.schema.fields)
            probs = probs * math.factorial(num_fields)

        return probs
