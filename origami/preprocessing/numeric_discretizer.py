"""Numeric discretization for high-cardinality numeric fields.

Provides per-field discretization using sklearn's KBinsDiscretizer.
This is an alternative to the continuous head implementation for handling
numeric values in a purely discrete token space.
"""

from collections import defaultdict
from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class NumericDiscretizer:
    """Discretize high-cardinality numeric fields into bins.

    Creates a separate KBinsDiscretizer for each numeric field that has
    more than `cat_threshold` distinct values. Fields with fewer distinct
    values are treated as categorical and pass through unchanged.

    Fields are identified by their JSON path (e.g., "price", "stats.score",
    "items.0.value"). Each field gets its own discretizer to handle
    different value ranges appropriately.

    Example:
        ```python
        discretizer = NumericDiscretizer(cat_threshold=20, n_bins=10)

        # Fit on training data
        train_data_binned = discretizer.fit_transform(train_data)

        # Transform test data (uses same bin edges)
        test_data_binned = discretizer.transform(test_data)

        # Check which fields were discretized
        print(discretizer.discretized_fields)
        ```

    Attributes:
        cat_threshold: Fields with <= this many distinct values pass through
        n_bins: Number of bins for discretization
        strategy: Binning strategy ('quantile', 'uniform', 'kmeans')
        discretizers: Dict mapping field path to fitted KBinsDiscretizer
        discretized_fields: Set of field paths that were discretized
        passthrough_fields: Set of field paths that pass through unchanged
    """

    def __init__(
        self,
        cat_threshold: int = 100,
        n_bins: int = 20,
        strategy: Literal["quantile", "uniform", "kmeans"] = "quantile",
    ):
        """Initialize discretizer.

        Args:
            cat_threshold: Maximum number of distinct values for a field
                to be treated as categorical (pass-through). Fields with
                more distinct values will be discretized. Default 100.
            n_bins: Number of bins for discretization. Default 20.
            strategy: Binning strategy. Options:
                - 'quantile': Each bin has approximately same number of samples
                - 'uniform': Each bin has same width
                - 'kmeans': Values clustered using k-means
                Default is 'quantile'.
        """
        if cat_threshold < 1:
            raise ValueError(f"cat_threshold must be >= 1, got {cat_threshold}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if strategy not in ("quantile", "uniform", "kmeans"):
            raise ValueError(f"strategy must be 'quantile', 'uniform', or 'kmeans', got {strategy}")

        self.cat_threshold = cat_threshold
        self.n_bins = n_bins
        self.strategy = strategy

        # Fitted state
        self.discretizers: dict[str, KBinsDiscretizer] = {}
        self.discretized_fields: set[str] = set()
        self.passthrough_fields: set[str] = set()
        self._fitted = False

    def fit(self, data: list[dict]) -> "NumericDiscretizer":
        """Fit discretizers on training data.

        Analyzes all numeric fields, determines which need discretization,
        and fits a KBinsDiscretizer for each high-cardinality field.

        Args:
            data: List of JSON objects to fit on

        Returns:
            self (for method chaining)
        """
        # Collect all numeric values by field path
        field_values: dict[str, list[float]] = defaultdict(list)
        self._collect_numeric_values(data, field_values)

        # Determine which fields need discretization
        self.discretizers = {}
        self.discretized_fields = set()
        self.passthrough_fields = set()

        for path, values in field_values.items():
            unique_count = len(set(values))

            if unique_count > self.cat_threshold:
                # High cardinality - fit a discretizer
                # Adjust n_bins if we have fewer unique values than bins
                effective_bins = min(self.n_bins, unique_count)

                # Build kwargs for KBinsDiscretizer
                kwargs: dict = {
                    "n_bins": effective_bins,
                    "encode": "ordinal",
                    "strategy": self.strategy,
                    "subsample": None,  # Use all data for fitting
                }
                # Silence FutureWarning for quantile strategy
                if self.strategy == "quantile":
                    kwargs["quantile_method"] = "averaged_inverted_cdf"

                discretizer = KBinsDiscretizer(**kwargs)

                # Fit on the collected values
                values_array = np.array(values).reshape(-1, 1)
                discretizer.fit(values_array)

                self.discretizers[path] = discretizer
                self.discretized_fields.add(path)
            else:
                # Low cardinality - pass through as categorical
                self.passthrough_fields.add(path)

        self._fitted = True
        return self

    def transform(self, data: list[dict]) -> list[dict]:
        """Transform data using fitted discretizers.

        Args:
            data: List of JSON objects to transform

        Returns:
            List of transformed JSON objects (new objects, not mutated)

        Raises:
            RuntimeError: If fit() has not been called
        """
        if not self._fitted:
            raise RuntimeError("NumericDiscretizer must be fit before transform")

        return [self._transform_object(obj) for obj in data]

    def fit_transform(self, data: list[dict]) -> list[dict]:
        """Fit and transform in one step.

        Args:
            data: List of JSON objects to fit and transform

        Returns:
            List of transformed JSON objects
        """
        self.fit(data)
        return self.transform(data)

    def _collect_numeric_values(
        self,
        data: list[dict],
        field_values: dict[str, list[float]],
    ) -> None:
        """Collect all numeric values from data, organized by field path."""
        for obj in data:
            self._collect_from_value(obj, "", field_values)

    def _collect_from_value(
        self,
        value: Any,
        path: str,
        field_values: dict[str, list[float]],
    ) -> None:
        """Recursively collect numeric values from a JSON value."""
        if isinstance(value, dict):
            for key, val in value.items():
                new_path = f"{path}.{key}" if path else key
                self._collect_from_value(val, new_path, field_values)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                new_path = f"{path}.{i}" if path else str(i)
                self._collect_from_value(item, new_path, field_values)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # Numeric value - record it
            if path:  # Only record if we have a path
                field_values[path].append(float(value))

    def _transform_object(self, obj: dict) -> dict:
        """Transform a single JSON object."""
        return self._transform_value(obj, "")

    def _transform_value(self, value: Any, path: str) -> Any:
        """Recursively transform a JSON value."""
        if isinstance(value, dict):
            return {
                key: self._transform_value(val, f"{path}.{key}" if path else key)
                for key, val in value.items()
            }
        elif isinstance(value, list):
            return [
                self._transform_value(item, f"{path}.{i}" if path else str(i))
                for i, item in enumerate(value)
            ]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # Check if this field should be discretized
            if path in self.discretizers:
                discretizer = self.discretizers[path]
                bin_idx = int(discretizer.transform([[float(value)]])[0, 0])
                # Return bin center instead of string label
                edges = discretizer.bin_edges_[0]
                return float((edges[bin_idx] + edges[bin_idx + 1]) / 2)
            else:
                # Pass through unchanged
                return value
        else:
            # Non-numeric value - pass through
            return value

    def get_bin_edges(self, field_path: str) -> np.ndarray | None:
        """Get bin edges for a discretized field.

        Args:
            field_path: Path to the field (e.g., "price", "stats.score")

        Returns:
            Array of bin edges, or None if field is not discretized
        """
        if field_path in self.discretizers:
            return self.discretizers[field_path].bin_edges_[0]
        return None

    def get_bin_label(self, field_path: str, bin_idx: int) -> str | None:
        """Get a descriptive label for a bin.

        Args:
            field_path: Path to the field
            bin_idx: Bin index (0 to n_bins-1)

        Returns:
            Label like "[0.5, 1.5)" or None if field not discretized
        """
        edges = self.get_bin_edges(field_path)
        if edges is None:
            return None

        if bin_idx < 0 or bin_idx >= len(edges) - 1:
            return None

        low = edges[bin_idx]
        high = edges[bin_idx + 1]
        return f"[{low:.4g}, {high:.4g})"

    def summary(self) -> str:
        """Return a summary of the discretizer state.

        Returns:
            Human-readable summary string
        """
        if not self._fitted:
            return "NumericDiscretizer (not fitted)"

        lines = [
            f"NumericDiscretizer (cat_threshold={self.cat_threshold}, n_bins={self.n_bins}, strategy={self.strategy})",
            f"  Discretized fields ({len(self.discretized_fields)}):",
        ]

        for path in sorted(self.discretized_fields):
            edges = self.get_bin_edges(path)
            n_bins = len(edges) - 1 if edges is not None else 0
            lines.append(f"    {path}: {n_bins} bins")

        lines.append(f"  Pass-through fields ({len(self.passthrough_fields)}):")
        for path in sorted(self.passthrough_fields):
            lines.append(f"    {path}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize discretizer state for checkpointing.

        Returns:
            Dictionary containing all state needed to reconstruct the discretizer.
        """
        if not self._fitted:
            raise RuntimeError("Cannot serialize unfitted NumericDiscretizer")

        return {
            "cat_threshold": self.cat_threshold,
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "discretized_fields": list(self.discretized_fields),
            "passthrough_fields": list(self.passthrough_fields),
            "discretizers": {
                path: {
                    "bin_edges": discretizer.bin_edges_[0].tolist(),
                    "n_bins": int(discretizer.n_bins_[0]),
                }
                for path, discretizer in self.discretizers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NumericDiscretizer":
        """Reconstruct discretizer from serialized state.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed NumericDiscretizer instance.
        """
        discretizer = cls(
            cat_threshold=data["cat_threshold"],
            n_bins=data["n_bins"],
            strategy=data["strategy"],
        )
        discretizer.discretized_fields = set(data["discretized_fields"])
        discretizer.passthrough_fields = set(data["passthrough_fields"])
        discretizer._fitted = True

        for path, bin_data in data["discretizers"].items():
            # Create a fitted KBinsDiscretizer by setting internal attributes
            kbd = KBinsDiscretizer(
                n_bins=bin_data["n_bins"],
                encode="ordinal",
                strategy=data["strategy"],
            )
            kbd.bin_edges_ = np.array([np.array(bin_data["bin_edges"])])
            kbd.n_bins_ = np.array([bin_data["n_bins"]])
            kbd.n_features_in_ = 1
            discretizer.discretizers[path] = kbd

        return discretizer
