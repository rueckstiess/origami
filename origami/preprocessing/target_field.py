"""Target field preprocessing utilities.

Provides utilities for reordering JSON objects to place target fields last,
which is useful for prediction tasks where the model should see all context
before predicting the target value.
"""

from typing import Any


def move_target_last(obj: dict[str, Any], target_key: str) -> dict[str, Any]:
    """Move target key (and value) to last position in object.

    Supports nested keys via dot notation (e.g., "foo.bar"):
    - Moves "foo" last at root level
    - Moves "bar" last within "foo" sub-object

    This ensures the model sees all other fields before predicting the target,
    maximizing the context available for prediction.

    If the target key doesn't exist, it is inserted with None as the value.
    This is useful for creating embeddings for prediction where the target
    value is unknown. For nested keys, intermediate dicts are created as needed.

    Args:
        obj: Input JSON object (not mutated)
        target_key: Dot-separated key path (e.g., "foo" or "foo.bar.baz")

    Returns:
        New dict with target key moved/inserted at end at each level

    Example:
        >>> obj = {"a": 1, "b": 2, "c": 3}
        >>> move_target_last(obj, "a")
        {"b": 2, "c": 3, "a": 1}

        >>> obj = {"x": {"p": 1, "q": 2}, "y": 3}
        >>> move_target_last(obj, "x.p")
        {"y": 3, "x": {"q": 2, "p": 1}}

        >>> obj = {"a": 1}
        >>> move_target_last(obj, "b")
        {"a": 1, "b": None}

        >>> obj = {"a": 1}
        >>> move_target_last(obj, "x.y")
        {"a": 1, "x": {"y": None}}
    """
    if not target_key:
        raise ValueError("target_key cannot be empty")

    parts = target_key.split(".")
    return _move_key_last_recursive(obj, parts)


def _move_key_last_recursive(obj: dict[str, Any], key_parts: list[str]) -> dict[str, Any]:
    """Recursively move key to last position at each level.

    If the key doesn't exist, it is inserted with None as the value.
    For nested keys, intermediate dicts are created as needed.

    Args:
        obj: Current object level
        key_parts: Remaining key path parts

    Returns:
        New dict with key moved/inserted at end
    """
    if not key_parts:
        return obj

    current_key = key_parts[0]
    remaining_parts = key_parts[1:]

    # Build new dict with current_key last
    result = {}

    # Get target value, defaulting to empty dict if there are remaining parts, else None
    if current_key in obj:
        target_value = obj[current_key]
    elif remaining_parts:
        # Need to create nested structure
        target_value = {}
    else:
        # Leaf key - insert None as placeholder
        target_value = None

    # Add all other keys first
    for key, value in obj.items():
        if key != current_key:
            result[key] = value

    # Process nested levels if there are remaining parts
    if remaining_parts:
        if not isinstance(target_value, dict):
            raise KeyError(
                f"Cannot access '{'.'.join(remaining_parts)}' in non-dict value at '{current_key}'"
            )
        target_value = _move_key_last_recursive(target_value, remaining_parts)

    # Add target key last
    result[current_key] = target_value

    return result
