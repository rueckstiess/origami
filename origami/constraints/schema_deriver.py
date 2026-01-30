"""Derive JSON Schema from training data.

Analyzes a collection of JSON objects and produces a JSON Schema (draft 2020-12
subset) that describes the observed data structure. This schema can then be used
by SchemaPDA to constrain model outputs during training and inference.

Derivation logic per field:
- Types: Collect Python types → map to JSON Schema types
- Enums: If unique value count <= enum_threshold, add enum constraint
- Objects: Recurse into properties. Fields present in ALL objects → required.
  Sets additionalProperties: false to prevent hallucinated keys.
- Arrays: Recurse into items (union of element schemas). Record minItems/maxItems
- Numerics: Record minimum/maximum from observed values
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from origami.preprocessing.numeric_scaler import ScaledNumeric

# Python type → JSON Schema type mapping
_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}


class SchemaDeriver:
    """Derive a JSON Schema dict from training data.

    Analyzes field types, value distributions, and structure to produce
    a schema that captures the data's constraints.

    Args:
        enum_threshold: Fields with more than this many unique primitive
            values will not have an enum constraint. None (default) means
            no limit — always include enum regardless of cardinality.
    """

    def __init__(self, enum_threshold: int | None = None):
        if enum_threshold is not None and enum_threshold < 1:
            raise ValueError(f"enum_threshold must be >= 1 or None, got {enum_threshold}")
        self.enum_threshold = enum_threshold

    def derive(self, data: list[dict]) -> dict:
        """Derive a JSON Schema from a list of JSON objects.

        Args:
            data: List of JSON objects (dicts) to analyze.

        Returns:
            A JSON Schema dict describing the observed data structure.

        Raises:
            ValueError: If data is empty or contains non-dict items.
        """
        if not data:
            raise ValueError("Cannot derive schema from empty data")

        if not all(isinstance(obj, dict) for obj in data):
            raise ValueError("All items in data must be dicts")

        return self._derive_object_schema(data)

    def _derive_object_schema(self, objects: list[dict]) -> dict:
        """Derive schema for a list of objects (all assumed to be dicts).

        Collects all keys across objects, determines required keys,
        and recurses into each field's values.
        """
        # Collect values per key
        key_values: dict[str, list[Any]] = defaultdict(list)
        key_presence: dict[str, int] = defaultdict(int)

        for obj in objects:
            for key, value in obj.items():
                key_values[key].append(value)
                key_presence[key] += 1

        total = len(objects)
        properties: dict[str, dict] = {}
        required: list[str] = []

        for key in sorted(key_values.keys()):
            values = key_values[key]
            properties[key] = self._derive_value_schema(values)

            # Field present in ALL objects → required
            if key_presence[key] == total:
                required.append(key)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required

        return schema

    def _derive_value_schema(self, values: list[Any]) -> dict:
        """Derive schema for a list of values at a given field.

        Handles mixed types, nested objects, arrays, and primitives.
        """
        # Classify values by kind
        object_values: list[dict] = []
        array_values: list[list] = []
        primitive_values: list[Any] = []
        scaled_values: list[float] = []

        for v in values:
            if isinstance(v, dict):
                object_values.append(v)
            elif isinstance(v, list):
                array_values.append(v)
            elif isinstance(v, ScaledNumeric):
                scaled_values.append(v.value)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                primitive_values.append(v)
            # Skip other types silently

        sub_schemas: list[dict] = []

        # Handle objects
        if object_values:
            sub_schemas.append(self._derive_object_schema(object_values))

        # Handle arrays
        if array_values:
            sub_schemas.append(self._derive_array_schema(array_values))

        # Handle primitives
        if primitive_values:
            sub_schemas.append(self._derive_primitive_schema(primitive_values))

        # Handle scaled numerics — type + bounds only, no enum
        if scaled_values:
            sub_schemas.append(
                {
                    "type": "number",
                    "minimum": min(scaled_values),
                    "maximum": max(scaled_values),
                }
            )

        if not sub_schemas:
            # No recognized values — permissive schema
            return {}

        if len(sub_schemas) == 1:
            return sub_schemas[0]

        # Multiple types — merge into a combined schema
        return self._merge_schemas(sub_schemas)

    def _derive_array_schema(self, arrays: list[list]) -> dict:
        """Derive schema for array values.

        Produces items schema from union of all elements, plus
        minItems/maxItems from observed lengths. Detects uniqueItems
        when all observed arrays contain no duplicate elements.
        """
        schema: dict[str, Any] = {"type": "array"}

        # Collect all elements across all arrays
        all_elements: list[Any] = []
        lengths: list[int] = []
        all_unique = True

        for arr in arrays:
            lengths.append(len(arr))
            all_elements.extend(arr)
            # Check if this array has all unique primitive elements
            # Only check arrays with >1 element (single-element arrays are trivially unique)
            if len(arr) > 1 and all_unique:
                try:
                    if len(set(arr)) != len(arr):
                        all_unique = False
                except TypeError:
                    # Unhashable elements (dicts, lists) — skip uniqueness check
                    all_unique = False

        # Derive items schema from all elements
        if all_elements:
            schema["items"] = self._derive_value_schema(all_elements)

        # Record observed length bounds
        if lengths:
            schema["minItems"] = min(lengths)
            schema["maxItems"] = max(lengths)

        # Mark as uniqueItems if all observed arrays have unique elements
        if all_unique and all_elements:
            schema["uniqueItems"] = True

        return schema

    def _derive_primitive_schema(self, values: list[Any]) -> dict:
        """Derive schema for primitive values (str, int, float, bool, None).

        Determines JSON type(s), enum, and numeric bounds.
        """
        # Collect JSON types (booleans must be checked before int since
        # isinstance(True, int) is True in Python)
        json_types: set[str] = set()
        numeric_values: list[int | float] = []

        for v in values:
            if isinstance(v, bool):
                json_types.add("boolean")
            elif isinstance(v, int):
                json_types.add("integer")
                numeric_values.append(v)
            elif isinstance(v, float):
                json_types.add("number")
                numeric_values.append(v)
            elif isinstance(v, str):
                json_types.add("string")
            elif v is None:
                json_types.add("null")

        # If we have both integer and number, consolidate to number
        if "integer" in json_types and "number" in json_types:
            json_types.discard("integer")

        schema: dict[str, Any] = {}

        # Set type
        if len(json_types) == 1:
            schema["type"] = next(iter(json_types))
        elif len(json_types) > 1:
            schema["type"] = sorted(json_types)

        # Enum: include unless threshold is set and exceeded
        unique_values = set()
        for v in values:
            unique_values.add(v)

        include_enum = self.enum_threshold is None or len(unique_values) <= self.enum_threshold
        if include_enum:
            # Sort for deterministic output (handle mixed types gracefully)
            enum_list = _sort_mixed(list(unique_values))
            schema["enum"] = enum_list

        # Numeric bounds
        if numeric_values:
            schema["minimum"] = min(numeric_values)
            schema["maximum"] = max(numeric_values)

        return schema

    def _merge_schemas(self, schemas: list[dict]) -> dict:
        """Merge multiple sub-schemas into one.

        Combines types and preserves nested structure.
        When object and non-object schemas are merged, uses a combined
        type list and preserves the object's properties.
        """
        # Collect all types
        all_types: list[str] = []
        merged: dict[str, Any] = {}

        for schema in schemas:
            schema_type = schema.get("type")
            if isinstance(schema_type, list):
                all_types.extend(schema_type)
            elif schema_type:
                all_types.append(schema_type)

            # Merge properties from object schemas
            if "properties" in schema:
                if "properties" not in merged:
                    merged["properties"] = {}
                merged["properties"].update(schema["properties"])

            # Merge items from array schemas
            if "items" in schema:
                merged["items"] = schema["items"]

            # Merge numeric bounds (widen)
            if "minimum" in schema:
                if "minimum" not in merged:
                    merged["minimum"] = schema["minimum"]
                else:
                    merged["minimum"] = min(merged["minimum"], schema["minimum"])
            if "maximum" in schema:
                if "maximum" not in merged:
                    merged["maximum"] = schema["maximum"]
                else:
                    merged["maximum"] = max(merged["maximum"], schema["maximum"])

            # Merge array bounds (widen)
            if "minItems" in schema:
                if "minItems" not in merged:
                    merged["minItems"] = schema["minItems"]
                else:
                    merged["minItems"] = min(merged["minItems"], schema["minItems"])
            if "maxItems" in schema:
                if "maxItems" not in merged:
                    merged["maxItems"] = schema["maxItems"]
                else:
                    merged["maxItems"] = max(merged["maxItems"], schema["maxItems"])

            # Merge uniqueItems (intersection — unique only if unique in ALL)
            if "uniqueItems" in schema:
                if "uniqueItems" not in merged:
                    merged["uniqueItems"] = schema["uniqueItems"]
                else:
                    merged["uniqueItems"] = merged["uniqueItems"] and schema["uniqueItems"]

            # Merge additionalProperties (AND — false only if false in ALL)
            if "additionalProperties" in schema:
                if "additionalProperties" not in merged:
                    merged["additionalProperties"] = schema["additionalProperties"]
                else:
                    # Both must be False for result to be False
                    merged["additionalProperties"] = (
                        merged["additionalProperties"] or schema["additionalProperties"]
                    )

            # Merge required (intersection — required only if required in ALL)
            if "required" in schema:
                if "required" not in merged:
                    merged["required"] = set(schema["required"])
                else:
                    merged["required"] = merged["required"] & set(schema["required"])

            # Merge enum (union)
            if "enum" in schema:
                if "enum" not in merged:
                    merged["enum"] = set()
                merged["enum"].update(schema["enum"])

        # Deduplicate types, consolidate integer+number → number
        unique_types = list(dict.fromkeys(all_types))
        if "integer" in unique_types and "number" in unique_types:
            unique_types.remove("integer")

        if len(unique_types) == 1:
            merged["type"] = unique_types[0]
        elif len(unique_types) > 1:
            merged["type"] = sorted(unique_types)

        # Convert required set back to sorted list
        if "required" in merged:
            req = merged["required"]
            if isinstance(req, set):
                merged["required"] = sorted(req) if req else None
                if merged["required"] is None:
                    del merged["required"]

        # Convert enum set back to sorted list
        if "enum" in merged:
            merged["enum"] = _sort_mixed(list(merged["enum"]))

        return merged


def _sort_mixed(values: list[Any]) -> list[Any]:
    """Sort a list of mixed types for deterministic output.

    Groups by type, then sorts within each group. Order:
    None, bool, int/float (numeric), str.
    """
    nones: list[None] = []
    bools: list[bool] = []
    numbers: list[int | float] = []
    strings: list[str] = []
    others: list[Any] = []

    for v in values:
        if v is None:
            nones.append(v)
        elif isinstance(v, bool):
            bools.append(v)
        elif isinstance(v, (int, float)):
            numbers.append(v)
        elif isinstance(v, str):
            strings.append(v)
        else:
            others.append(v)

    result: list[Any] = []
    result.extend(nones)
    result.extend(sorted(bools))
    result.extend(sorted(numbers))
    result.extend(sorted(strings))
    result.extend(others)
    return result
