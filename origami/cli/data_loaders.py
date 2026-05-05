"""Data loading utilities for the Origami CLI.

Supports multiple data formats:
- CSV files (*.csv)
- JSON files (*.json) - array of objects
- JSONL files (*.jsonl) - one object per line
- MongoDB collections (mongodb:// URI)
"""

from __future__ import annotations

import csv
import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from typing import Any


class DataFormat(Enum):
    """Supported data formats."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    MONGODB = "mongodb"


def detect_format(path: str) -> DataFormat:
    """Auto-detect data format from path extension or prefix.

    Args:
        path: File path or MongoDB URI

    Returns:
        Detected DataFormat

    Raises:
        click.BadParameter: If format cannot be detected
    """
    if path.startswith("mongodb://") or path.startswith("mongodb+srv://"):
        return DataFormat.MONGODB

    path_lower = path.lower()
    if path_lower.endswith(".csv"):
        return DataFormat.CSV
    elif path_lower.endswith(".json"):
        return DataFormat.JSON
    elif path_lower.endswith(".jsonl"):
        return DataFormat.JSONL
    else:
        raise click.BadParameter(
            f"Cannot detect format from '{path}'. "
            "Use .csv, .json, .jsonl extension or mongodb:// URI."
        )


def load_csv(path: str) -> list[dict[str, Any]]:
    """Load data from a CSV file.

    Args:
        path: Path to CSV file

    Returns:
        List of dictionaries, one per row
    """
    data = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to numbers where possible
            converted = {}
            for key, value in row.items():
                converted[key] = _convert_value(value)
            data.append(converted)
    return data


def _convert_value(value: str) -> Any:
    """Convert a string value to appropriate Python type."""
    if value == "":
        return None

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Keep as string
    return value


def get_nested_value(obj: dict[str, Any], dotted_key: str) -> Any:
    """Get a value from a nested dict using dot notation.

    Args:
        obj: The dictionary to traverse
        dotted_key: Key path with dots (e.g., "foo.bar.baz")

    Returns:
        The value at the path

    Raises:
        KeyError: If the path doesn't exist
    """
    keys = dotted_key.split(".")
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(dotted_key)
        current = current[key]
    return current


def set_nested_value(obj: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation, creating intermediate dicts.

    Args:
        obj: The dictionary to modify
        dotted_key: Key path with dots (e.g., "foo.bar.baz")
        value: The value to set
    """
    keys = dotted_key.split(".")
    current = obj
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _delete_nested_key(obj: dict[str, Any], dotted_key: str) -> None:
    """Delete a nested key from a dictionary using dot notation.

    Args:
        obj: The dictionary to modify in place
        dotted_key: Key path with dots (e.g., "foo.bar")
    """
    keys = dotted_key.split(".")
    current = obj
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            return  # Path doesn't exist, nothing to delete
        current = current[key]

    if isinstance(current, dict):
        current.pop(keys[-1], None)


def parse_projection(projection_json: str) -> dict[str, int]:
    """Parse a MongoDB-style projection JSON string.

    Args:
        projection_json: JSON string like '{"a": 1, "b": 1}' or '{"x": 0}'

    Returns:
        Parsed projection dict mapping field names to 0 or 1

    Raises:
        click.BadParameter: If JSON is invalid or mixes inclusion/exclusion
    """
    try:
        projection = json.loads(projection_json)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid projection JSON: {e}") from e

    if not isinstance(projection, dict):
        raise click.BadParameter("Projection must be a JSON object")

    # Validate all values are 0 or 1
    values = set()
    for key, val in projection.items():
        if val not in (0, 1):
            raise click.BadParameter(
                f"Projection values must be 0 or 1, got {val!r} for key '{key}'"
            )
        values.add(val)

    # Check for mixed inclusion/exclusion
    if len(values) > 1:
        raise click.BadParameter("Cannot mix inclusion (1) and exclusion (0) in projection")

    return projection


def apply_projection(obj: dict[str, Any], projection: dict[str, int]) -> dict[str, Any]:
    """Apply a MongoDB-style projection to a dictionary.

    Args:
        obj: The original dictionary
        projection: Mapping of dotted keys to 0 (exclude) or 1 (include)

    Returns:
        New dictionary with projection applied
    """
    if not projection:
        return obj

    # Determine if inclusion (1) or exclusion (0) mode
    mode = next(iter(projection.values()))

    if mode == 1:
        # Inclusion mode: only include specified fields
        result: dict[str, Any] = {}
        for dotted_key in projection:
            try:
                value = get_nested_value(obj, dotted_key)
                set_nested_value(result, dotted_key, value)
            except KeyError:
                # Field not present in this object - skip silently
                pass
        return result
    else:
        # Exclusion mode: include all except specified fields
        import copy

        result = copy.deepcopy(obj)
        for dotted_key in projection:
            _delete_nested_key(result, dotted_key)
        return result


def load_json(path: str) -> list[dict[str, Any]]:
    """Load data from a JSON file (array of objects).

    Args:
        path: Path to JSON file

    Returns:
        List of dictionaries
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise click.BadParameter(f"JSON file must contain an array, got {type(data).__name__}")

    return data


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load data from a JSONL file (one object per line).

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Remove MongoDB _id field if present
                obj.pop("_id", None)
                data.append(obj)
            except json.JSONDecodeError as e:
                raise click.BadParameter(f"Invalid JSON on line {line_num}: {e}") from e
    return data


def load_mongodb(
    uri: str,
    db: str,
    collection: str,
    *,
    skip: int = 0,
    limit: int = 0,
    projection: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Load data from a MongoDB collection.

    Args:
        uri: MongoDB connection URI
        db: Database name
        collection: Collection name
        skip: Number of documents to skip (default: 0)
        limit: Maximum documents to return, 0 = unlimited (default: 0)
        projection: MongoDB projection dict (default: None)

    Returns:
        List of dictionaries
    """
    try:
        from pymongo import MongoClient
    except ImportError as e:
        raise click.ClickException(
            "MongoDB support requires pymongo. Install with: pip install origami[mongodb]"
        ) from e

    client = MongoClient(uri)
    database = client[db]
    coll = database[collection]

    # Build MongoDB projection - always exclude _id
    mongo_projection: dict[str, Any] | None = None
    if projection:
        mongo_projection = dict(projection)
        mongo_projection["_id"] = 0
    else:
        mongo_projection = {"_id": 0}

    # Build cursor with skip/limit/projection
    cursor = coll.find(projection=mongo_projection)
    if skip > 0:
        cursor = cursor.skip(skip)
    if limit > 0:
        cursor = cursor.limit(limit)

    data = list(cursor)
    client.close()
    return data


def load_data(
    path: str,
    db: str | None = None,
    collection: str | None = None,
    *,
    skip: int = 0,
    limit: int = 0,
    project: str | None = None,
) -> list[dict[str, Any]]:
    """Load data from any supported source with auto-detection.

    Args:
        path: File path or MongoDB URI
        db: Database name (required for MongoDB)
        collection: Collection name (required for MongoDB)
        skip: Number of samples to skip (default: 0)
        limit: Maximum samples to return, 0 = unlimited (default: 0)
        project: MongoDB-style projection JSON string (default: None)

    Returns:
        List of dictionaries
    """
    # Parse projection if provided
    projection = parse_projection(project) if project else None

    fmt = detect_format(path)

    if fmt == DataFormat.MONGODB:
        if not db:
            raise click.BadParameter("--db is required for MongoDB data source")
        if not collection:
            raise click.BadParameter("-c/--collection is required for MongoDB data source")
        return load_mongodb(path, db, collection, skip=skip, limit=limit, projection=projection)

    # Validate file exists
    if not Path(path).exists():
        raise click.BadParameter(f"File not found: {path}")

    # Load all data from file
    if fmt == DataFormat.CSV:
        data = load_csv(path)
    elif fmt == DataFormat.JSON:
        data = load_json(path)
    elif fmt == DataFormat.JSONL:
        data = load_jsonl(path)
    else:
        raise click.BadParameter(f"Unsupported format: {fmt}")

    # Apply skip/limit via slicing (for file formats)
    if skip > 0:
        data = data[skip:]
    if limit > 0:
        data = data[:limit]

    # Apply projection (for file formats)
    if projection:
        data = [apply_projection(obj, projection) for obj in data]

    return data
