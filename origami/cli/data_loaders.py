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


def load_mongodb(uri: str, db: str, collection: str) -> list[dict[str, Any]]:
    """Load data from a MongoDB collection.

    Args:
        uri: MongoDB connection URI
        db: Database name
        collection: Collection name

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

    data = []
    for doc in coll.find():
        # Remove MongoDB _id field
        doc.pop("_id", None)
        data.append(doc)

    client.close()
    return data


def load_data(
    path: str,
    db: str | None = None,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    """Load data from any supported source with auto-detection.

    Args:
        path: File path or MongoDB URI
        db: Database name (required for MongoDB)
        collection: Collection name (required for MongoDB)

    Returns:
        List of dictionaries
    """
    fmt = detect_format(path)

    if fmt == DataFormat.MONGODB:
        if not db:
            raise click.BadParameter("--db is required for MongoDB data source")
        if not collection:
            raise click.BadParameter("-c/--collection is required for MongoDB data source")
        return load_mongodb(path, db, collection)

    # Validate file exists
    if not Path(path).exists():
        raise click.BadParameter(f"File not found: {path}")

    if fmt == DataFormat.CSV:
        return load_csv(path)
    elif fmt == DataFormat.JSON:
        return load_json(path)
    elif fmt == DataFormat.JSONL:
        return load_jsonl(path)
    else:
        raise click.BadParameter(f"Unsupported format: {fmt}")
