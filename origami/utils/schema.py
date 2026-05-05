"""Utilities for displaying JSON Schema dicts."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

MAX_ENUM_DISPLAY = 5


def truncate_schema_enums(schema: dict, max_display: int = MAX_ENUM_DISPLAY) -> dict:
    """Return a copy of a JSON Schema with long enum lists truncated.

    Enum lists with more than *max_display* entries are shortened to the
    first *max_display* values followed by a ``"... + N more"`` sentinel.

    The original schema dict is not modified.
    """
    schema = deepcopy(schema)
    _truncate_enums_inplace(schema, max_display)
    return schema


def format_schema(schema: dict, max_enum_display: int = MAX_ENUM_DISPLAY) -> str:
    """Format a JSON Schema as indented JSON with truncated enum lists."""
    truncated = truncate_schema_enums(schema, max_enum_display)
    return json.dumps(truncated, indent=2)


def _truncate_enums_inplace(node: Any, max_display: int) -> None:
    """Recursively truncate ``enum`` lists inside *node* in-place."""
    if not isinstance(node, dict):
        return

    if "enum" in node:
        enum = node["enum"]
        if isinstance(enum, list) and len(enum) > max_display:
            remaining = len(enum) - max_display
            node["enum"] = enum[:max_display] + [f"... + {remaining} more"]

    # Recurse into sub-schemas
    if "properties" in node and isinstance(node["properties"], dict):
        for prop_schema in node["properties"].values():
            _truncate_enums_inplace(prop_schema, max_display)

    if "items" in node and isinstance(node["items"], dict):
        _truncate_enums_inplace(node["items"], max_display)
