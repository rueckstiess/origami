"""Numeric scaling for high-cardinality numeric fields.

Provides per-field scaling using sklearn's StandardScaler for use with
the continuous head (Mixture of Gaussians output). High-cardinality numeric
fields are replaced with ScaledNumeric markers that the tokenizer converts
to NUM tokens with associated scaled values.
"""

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


class ScaledNumeric:
    """Marker class for scaled numeric values.

    When the tokenizer encounters a ScaledNumeric, it emits a NUM token
    and records the scaled value for the continuous head.

    Attributes:
        value: The scaled numeric value (typically z-score normalized)
    """

    __slots__ = ("value",)

    def __init__(self, value: float):
        """Initialize with scaled value.

        Args:
            value: The scaled numeric value
        """
        self.value = value

    def __repr__(self) -> str:
        return f"ScaledNumeric({self.value:.4f})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ScaledNumeric):
            return self.value == other.value
        return False


class NumericScaler:
    """Scale high-cardinality numeric fields for continuous head.

    Creates a separate StandardScaler for each numeric field that has
    more than `cat_threshold` distinct values. Fields with fewer distinct
    values are treated as categorical and pass through unchanged.

    Fields are identified by their JSON path (e.g., "price", "stats.score",
    "items.0.value"). Each field gets its own scaler to handle different
    value ranges appropriately.

    Example:
        ```python
        scaler = NumericScaler(cat_threshold=100)

        # Fit on training data
        train_data_scaled = scaler.fit_transform(train_data)

        # Transform test data (uses same scaling)
        test_data_scaled = scaler.transform(test_data)

        # Check which fields were scaled
        print(scaler.scaled_fields)

        # During inference, convert back to original scale
        original_value = scaler.inverse_transform_value("price", scaled_value)
        ```

    Attributes:
        cat_threshold: Fields with <= this many distinct values pass through
        scalers: Dict mapping field path to fitted StandardScaler
        scaled_fields: Set of field paths that were scaled
        passthrough_fields: Set of field paths that pass through unchanged
    """

    def __init__(self, cat_threshold: int = 100):
        """Initialize scaler.

        Args:
            cat_threshold: Maximum number of distinct values for a field
                to be treated as categorical (pass-through). Fields with
                more distinct values will be scaled. Default 100.
        """
        if cat_threshold < 1:
            raise ValueError(f"cat_threshold must be >= 1, got {cat_threshold}")

        self.cat_threshold = cat_threshold

        # Fitted state
        self.scalers: dict[str, StandardScaler] = {}
        self.scaled_fields: set[str] = set()
        self.passthrough_fields: set[str] = set()
        self._fitted = False

    def fit(self, data: list[dict]) -> "NumericScaler":
        """Fit scalers on training data.

        Analyzes all numeric fields, determines which need scaling,
        and fits a StandardScaler for each high-cardinality field.

        Args:
            data: List of JSON objects to fit on

        Returns:
            self (for method chaining)
        """
        # Collect all numeric values by field path
        field_values: dict[str, list[float]] = defaultdict(list)
        self._collect_numeric_values(data, field_values)

        # Determine which fields need scaling
        self.scalers = {}
        self.scaled_fields = set()
        self.passthrough_fields = set()

        for path, values in field_values.items():
            unique_count = len(set(values))

            if unique_count > self.cat_threshold:
                # High cardinality - fit a scaler
                scaler = StandardScaler()
                values_array = np.array(values).reshape(-1, 1)
                scaler.fit(values_array)

                self.scalers[path] = scaler
                self.scaled_fields.add(path)
            else:
                # Low cardinality - pass through as categorical
                self.passthrough_fields.add(path)

        self._fitted = True
        return self

    def transform(self, data: list[dict]) -> list[dict]:
        """Transform data using fitted scalers.

        High-cardinality numeric values are replaced with ScaledNumeric
        markers. Low-cardinality values pass through unchanged.

        Args:
            data: List of JSON objects to transform

        Returns:
            List of transformed JSON objects (new objects, not mutated)

        Raises:
            RuntimeError: If fit() has not been called
        """
        if not self._fitted:
            raise RuntimeError("NumericScaler must be fit before transform")

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

    def inverse_transform_value(self, field_path: str, scaled_value: float) -> float:
        """Convert a scaled value back to original scale.

        Args:
            field_path: Path to the field (e.g., "price", "stats.score")
            scaled_value: The scaled value to convert back

        Returns:
            Original-scale value

        Raises:
            KeyError: If field_path is not a scaled field
            RuntimeError: If fit() has not been called
        """
        if not self._fitted:
            raise RuntimeError("NumericScaler must be fit before inverse_transform")

        if field_path not in self.scalers:
            raise KeyError(f"Field '{field_path}' is not a scaled field")

        scaler = self.scalers[field_path]
        original = scaler.inverse_transform([[scaled_value]])[0, 0]
        return float(original)

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
            # Check if this field should be scaled
            if path in self.scalers:
                scaler = self.scalers[path]
                scaled = scaler.transform([[float(value)]])[0, 0]
                return ScaledNumeric(float(scaled))
            else:
                # Pass through unchanged
                return value
        else:
            # Non-numeric value - pass through
            return value

    def get_scaler_stats(self, field_path: str) -> dict[str, float] | None:
        """Get mean and std for a scaled field.

        Args:
            field_path: Path to the field (e.g., "price", "stats.score")

        Returns:
            Dict with 'mean' and 'std' keys, or None if field is not scaled
        """
        if field_path in self.scalers:
            scaler = self.scalers[field_path]
            return {
                "mean": float(scaler.mean_[0]),
                "std": float(scaler.scale_[0]),
            }
        return None

    def summary(self) -> str:
        """Return a summary of the scaler state.

        Returns:
            Human-readable summary string
        """
        if not self._fitted:
            return "NumericScaler (not fitted)"

        lines = [
            f"NumericScaler (cat_threshold={self.cat_threshold})",
            f"  Scaled fields ({len(self.scaled_fields)}):",
        ]

        for path in sorted(self.scaled_fields):
            stats = self.get_scaler_stats(path)
            if stats:
                lines.append(f"    {path}: mean={stats['mean']:.4g}, std={stats['std']:.4g}")

        lines.append(f"  Pass-through fields ({len(self.passthrough_fields)}):")
        for path in sorted(self.passthrough_fields):
            lines.append(f"    {path}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize scaler state for checkpointing.

        Returns:
            Dictionary containing all state needed to reconstruct the scaler.
        """
        if not self._fitted:
            raise RuntimeError("Cannot serialize unfitted NumericScaler")

        return {
            "cat_threshold": self.cat_threshold,
            "scaled_fields": list(self.scaled_fields),
            "passthrough_fields": list(self.passthrough_fields),
            "scalers": {
                path: {
                    "mean": float(scaler.mean_[0]),
                    "scale": float(scaler.scale_[0]),
                }
                for path, scaler in self.scalers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NumericScaler":
        """Reconstruct scaler from serialized state.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed NumericScaler instance.
        """
        scaler = cls(cat_threshold=data["cat_threshold"])
        scaler.scaled_fields = set(data["scaled_fields"])
        scaler.passthrough_fields = set(data["passthrough_fields"])
        scaler._fitted = True

        for path, stats in data["scalers"].items():
            ss = StandardScaler()
            ss.mean_ = np.array([stats["mean"]])
            ss.scale_ = np.array([stats["scale"]])
            ss.var_ = np.array([stats["scale"] ** 2])
            ss.n_features_in_ = 1
            ss.n_samples_seen_ = 1  # Dummy value
            scaler.scalers[path] = ss

        return scaler
