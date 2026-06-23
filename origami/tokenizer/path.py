"""Path representation for JSON structure traversal.

Paths track the location of tokens within a JSON document using a sequence
of KeyElement (for object keys) and IndexElement (for array indices).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyElement:
    """A key in a JSON object path.

    Example:
        For {"user": {"name": "Alice"}}, the path to "Alice" includes
        KeyElement("user") and KeyElement("name").
    """

    key: str

    def __repr__(self) -> str:
        return f"KeyElement({self.key!r})"


@dataclass(frozen=True)
class IndexElement:
    """An index in a JSON array path.

    Example:
        For {"items": ["a", "b", "c"]}, the path to "b" includes
        KeyElement("items") and IndexElement(1).
    """

    index: int

    def __repr__(self) -> str:
        return f"IndexElement({self.index})"


# Type alias for a path through a JSON structure
PathElement = KeyElement | IndexElement
Path = tuple[PathElement, ...]


def path_to_string(path: Path) -> str:
    """Convert a path to a human-readable string representation.

    Example:
        >>> path_to_string((KeyElement("user"), KeyElement("name")))
        'user.name'
        >>> path_to_string((KeyElement("items"), IndexElement(1)))
        'items[1]'
        >>> path_to_string(())
        '<root>'
    """
    if not path:
        return "<root>"

    parts = []
    for element in path:
        if isinstance(element, KeyElement):
            if parts:
                parts.append(f".{element.key}")
            else:
                parts.append(element.key)
        else:  # IndexElement
            parts.append(f"[{element.index}]")

    return "".join(parts)


def normalize_path(path: Path) -> str:
    """Convert a path to a normalized schema-path string.

    Array indices are replaced with ``*`` wildcards so that all elements at the
    same array position share a single key. This is the canonical keying used by
    schema constraints and array-length normalization, so any component that
    needs to match those keys must use this function.

    Examples:
        >>> normalize_path(())
        ''
        >>> normalize_path((KeyElement("name"),))
        'name'
        >>> normalize_path((KeyElement("items"), IndexElement(0)))
        'items.*'
        >>> normalize_path((KeyElement("items"), IndexElement(2), KeyElement("price")))
        'items.*.price'
    """
    parts: list[str] = []
    for element in path:
        if isinstance(element, KeyElement):
            parts.append(element.key)
        elif isinstance(element, IndexElement):
            parts.append("*")
    return ".".join(parts)
