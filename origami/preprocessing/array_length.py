"""Array-length normalization derived from data.

When ``model_array_lengths=True``, the continuous head regresses a *normalized*
array length ``length / norm`` (with bin width ``1 / norm``). The divisor ``norm``
must be identical at training and inference time, and must not depend on whether
schema masking is enabled — otherwise predicted lengths are mis-scaled.

The source of truth is a per-array-path maximum length derived from the training
data. The same map is used by the collator (training) and the generator
(inference), keyed by :func:`origami.tokenizer.path.normalize_path`.
"""

from __future__ import annotations

from typing import Any

from origami.tokenizer.path import Path, normalize_path


def derive_array_max_lengths(data: list[dict]) -> dict[str, int]:
    """Derive the maximum observed array length per normalized path.

    Walks every object and records, for each list, the largest length seen
    under its normalized path (array indices collapsed to ``*``). Nested arrays
    are keyed by the wildcard path of their container, matching the keys that
    the collator and generator look up at ARRAY_START positions.

    Args:
        data: Training data as a list of JSON-like dictionaries.

    Returns:
        Mapping from normalized path string to maximum observed array length.
        Empty if the data contains no arrays.
    """
    out: dict[str, int] = {}

    def _walk(value: Any, path_str: str) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                child = f"{path_str}.{key}" if path_str else key
                _walk(sub, child)
        elif isinstance(value, list):
            # Always record the path (even for empty arrays) so it is known;
            # the divisor is guarded to >= 1 at lookup time.
            out[path_str] = max(out.get(path_str, 0), len(value))
            child = f"{path_str}.*" if path_str else "*"
            for item in value:
                _walk(item, child)

    for obj in data:
        _walk(obj, "")

    return out


# Fractional headroom added above the observed max when normalizing array
# lengths. Keeps the largest observed length off the discretized-logistic upper
# absorbing boundary (which otherwise causes a sampling spike at the max). The
# valid sampling range is still capped at the observed max, so the headroom only
# affects the grid scale, never the range of generated lengths.
ARRAY_LENGTH_HEADROOM_FRAC = 0.1


def _headroom(max_len: int) -> int:
    """Integer headroom for a given observed max length (>= 1)."""
    return max(1, round(ARRAY_LENGTH_HEADROOM_FRAC * max_len))


def array_length_cap_for_key(key: str | None, max_lengths: dict[str, int]) -> int:
    """Return the maximum *valid* array length for a normalized path key.

    This is the largest length that may be sampled at the given path: the
    per-path observed maximum, with a global fallback when the key is absent.
    Always >= 1.

    Args:
        key: Normalized path string (see :func:`normalize_path`), or None.
        max_lengths: Per-path maximum lengths from :func:`derive_array_max_lengths`.
    """
    if not max_lengths:
        return 1
    n = max_lengths.get(key) if key is not None else None
    if n is None:
        n = max(max_lengths.values())
    return max(int(n), 1)


def array_length_norm_for_key(key: str | None, max_lengths: dict[str, int]) -> float:
    """Return the normalization divisor (grid scale) for a normalized path key.

    Equals the observed max (:func:`array_length_cap_for_key`) plus a small
    headroom, so the largest observed length maps below 1.0 rather than onto the
    upper absorbing boundary. The headroom is derived deterministically from the
    stored max, so training and inference always agree.

    Args:
        key: Normalized path string (see :func:`normalize_path`), or None.
        max_lengths: Per-path maximum lengths from :func:`derive_array_max_lengths`.

    Returns:
        Normalization divisor (> observed max, >= 1.0).
    """
    if not max_lengths:
        return 1.0
    cap = array_length_cap_for_key(key, max_lengths)
    return float(cap + _headroom(cap))


def array_length_norm(path: Path, max_lengths: dict[str, int]) -> float:
    """Return the normalization divisor (grid scale) for an array at ``path``.

    Args:
        path: Tokenizer path to the array (the ARRAY_START position).
        max_lengths: Per-path maximum lengths from :func:`derive_array_max_lengths`.

    Returns:
        Normalization divisor (> observed max, >= 1.0).
    """
    return array_length_norm_for_key(normalize_path(path), max_lengths)
