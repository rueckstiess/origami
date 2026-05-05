"""Schema-based post-processing for generated and predicted values.

Applies type coercion and value snapping based on a JSON Schema derived
from the original (unprocessed) training data. This corrects artifacts
introduced by numeric preprocessing (scaling, discretization) where
integer fields become floats and values don't match the original
distribution.

Post-processing operations (applied in order):
1. Clip to minimum/maximum bounds
2. Snap to nearest enum value (if enum constraint exists)
3. Round to integer (if type is "integer" and no enum)
"""

from __future__ import annotations

import bisect
from typing import Any


class SchemaPostProcessor:
    """Post-process generated/predicted values using original-data schema.

    Pre-indexes the schema by dot-separated field path for O(1) lookup.
    Array items use ``*`` wildcard (e.g., ``"items.*"``).

    Example:
        ```python
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "score": {"type": "number", "enum": [0.5, 1.0, 1.5, 2.0]},
            },
        }

        pp = SchemaPostProcessor(schema)
        pp.process_value(25.03, "age")    # -> 25
        pp.process_value(1.23, "score")   # -> 1.0
        pp.process_object({"age": 25.03, "score": 1.23})  # -> {"age": 25, "score": 1.0}
        ```
    """

    def __init__(self, schema: dict):
        self._schema = schema
        self._field_schemas: dict[str, dict] = {}
        # Pre-sorted numeric enums per path for O(log n) nearest-neighbor lookup
        self._sorted_enums: dict[str, list[float | int]] = {}
        self._compile_paths(schema, "")

    def _compile_paths(self, node: dict, path: str) -> None:
        """Recursively index field schemas by dot-separated path."""
        self._field_schemas[path] = node

        # Pre-sort numeric enums for O(log n) nearest-neighbor lookup
        schema_enum = node.get("enum")
        if schema_enum is not None:
            numeric_enums = sorted(
                e for e in schema_enum if isinstance(e, (int, float)) and not isinstance(e, bool)
            )
            if numeric_enums:
                self._sorted_enums[path] = numeric_enums

        if "properties" in node:
            for key, sub in node["properties"].items():
                sub_path = f"{path}.{key}" if path else key
                self._compile_paths(sub, sub_path)

        if "items" in node:
            sub_path = f"{path}.*" if path else "*"
            self._compile_paths(node["items"], sub_path)

    def process_value(self, value: Any, field_path: str) -> Any:
        """Post-process a single value based on its field's schema.

        Looks up the schema for ``field_path`` and applies corrections:
        bounds clipping, enum snapping, integer rounding.

        Args:
            value: The value to post-process.
            field_path: Dot-separated path (e.g., ``"age"``, ``"stats.score"``).

        Returns:
            The corrected value, or the original if no correction applies.
        """
        field_schema = self._field_schemas.get(field_path)
        if field_schema is None:
            return value

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return value

        schema_type = field_schema.get("type")
        minimum = field_schema.get("minimum")
        maximum = field_schema.get("maximum")

        # Step 1: Clip to bounds
        if minimum is not None and isinstance(minimum, (int, float)):
            value = max(value, minimum)
        if maximum is not None and isinstance(maximum, (int, float)):
            value = min(value, maximum)

        # Step 2: Snap to nearest enum value (if present)
        sorted_enums = self._sorted_enums.get(field_path)
        if sorted_enums is not None:
            idx = bisect.bisect_left(sorted_enums, value)
            # Check the closest of the two neighbors
            if idx == 0:
                return sorted_enums[0]
            if idx == len(sorted_enums):
                return sorted_enums[-1]
            before = sorted_enums[idx - 1]
            after = sorted_enums[idx]
            return before if (value - before) <= (after - value) else after

        # Step 3: Round to integer (if type is "integer")
        if _is_integer_type(schema_type):
            return int(round(value))

        return value

    def process_object(self, obj: dict, path: str = "") -> dict:
        """Recursively post-process all values in a generated object.

        Args:
            obj: Generated JSON object (dict).
            path: Base path prefix (empty for top-level).

        Returns:
            New dict with post-processed values.
        """
        result = {}
        for key, val in obj.items():
            sub_path = f"{path}.{key}" if path else key
            result[key] = self._process_recursive(val, sub_path)
        return result

    def _process_recursive(self, value: Any, path: str) -> Any:
        """Recursively process a value at the given path."""
        if isinstance(value, dict):
            return self.process_object(value, path)
        elif isinstance(value, list):
            # Array items share a single schema under the "*" wildcard
            item_path = f"{path}.*"
            return [self._process_recursive(item, item_path) for item in value]
        else:
            return self.process_value(value, path)


def _is_integer_type(schema_type: str | list[str] | None) -> bool:
    """Check if a schema type indicates integer.

    Returns True for ``"integer"`` or a type list containing ``"integer"``
    but not ``"number"`` (which subsumes integer).
    """
    if schema_type is None:
        return False
    if schema_type == "integer":
        return True
    if isinstance(schema_type, list):
        return "integer" in schema_type and "number" not in schema_type
    return False
