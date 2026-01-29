"""Numba-accelerated grammar mask computation.

Provides grammar mask computation using Numba JIT compilation.
When used with DataLoader workers (num_workers > 0), each worker
processes batches sequentially - the parallelism comes from multiple
workers, not from threading within each worker.

To avoid thread contention in DataLoader workers, the trainer's
worker_init_fn sets NUMBA_NUM_THREADS=1 per-worker (see trainer.py).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Stub decorators when numba not available
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

    prange = range  # type: ignore[misc]


# Stack content types (must match json_grammar.py)
STACK_EMPTY = 0
STACK_OBJECT = 1
STACK_ARRAY = 2


@njit(cache=True)
def _update_state(
    token: int,
    stack: np.ndarray,  # (max_depth,)
    depth: int,
    awaiting_value: int,  # bool as int
    seen_start: int,  # bool as int
    root_closed: int,  # bool as int
    ended: int,  # bool as int
    # Token IDs
    start_id: int,
    end_id: int,
    obj_start_id: int,
    obj_end_id: int,
    array_start_id: int,
    array_end_id: int,
    pad_id: int,
    key_ids: np.ndarray,
    value_ids: np.ndarray,
    max_depth: int,
) -> tuple[int, int, int, int, int]:
    """Update PDA state based on current token.

    Returns: (depth, awaiting_value, seen_start, root_closed, ended)
    Note: stack is modified in-place.
    """
    # Check token types
    is_start = token == start_id
    is_end = token == end_id
    is_obj_start = token == obj_start_id
    is_obj_end = token == obj_end_id
    is_array_start = token == array_start_id
    is_array_end = token == array_end_id
    is_pad = token == pad_id

    # Check if key or value (using linear search - fast for small arrays)
    is_key = False
    for kid in key_ids:
        if token == kid:
            is_key = True
            break

    is_value = False
    for vid in value_ids:
        if token == vid:
            is_value = True
            break

    # Skip PAD tokens
    if is_pad:
        return depth, awaiting_value, seen_start, root_closed, ended

    # START: mark seen_start
    if is_start:
        seen_start = 1

    # END: mark as ended
    if is_end:
        ended = 1

    # OBJ_START: push object onto stack
    if is_obj_start and depth < max_depth:
        stack[depth] = STACK_OBJECT
        depth += 1
        awaiting_value = 0

    # ARRAY_START: push array onto stack
    if is_array_start and depth < max_depth:
        stack[depth] = STACK_ARRAY
        depth += 1

    # OBJ_END: pop from stack
    if is_obj_end and depth > 0:
        if depth == 1:
            root_closed = 1
        depth -= 1

    # ARRAY_END: pop from stack
    if is_array_end and depth > 0:
        if depth == 1:
            root_closed = 1
        depth -= 1

    # Key: set awaiting_value
    if is_key:
        awaiting_value = 1

    # Value (primitive): clear awaiting_value
    if is_value:
        awaiting_value = 0

    # After closing a container, if parent is object, awaiting_value = 0
    if (is_obj_end or is_array_end) and depth > 0:
        if stack[depth - 1] == STACK_OBJECT:
            awaiting_value = 0

    return depth, awaiting_value, seen_start, root_closed, ended


@njit(cache=True)
def _get_valid_mask(
    stack: np.ndarray,  # (max_depth,)
    depth: int,
    awaiting_value: int,
    seen_start: int,
    root_closed: int,
    ended: int,
    # Token IDs
    start_id: int,
    end_id: int,
    obj_start_id: int,
    obj_end_id: int,
    array_start_id: int,
    array_end_id: int,
    pad_id: int,
    key_ids: np.ndarray,
    value_ids: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Get valid tokens mask based on current PDA state.

    Returns: (vocab_size,) boolean mask
    """
    valid = np.zeros(vocab_size, dtype=np.bool_)

    # Case 1: After END -> only PAD valid
    if ended:
        valid[pad_id] = True
        return valid

    # Case 2: Root closed -> only END valid
    if root_closed:
        valid[end_id] = True
        return valid

    # Case 3: Not seen START -> only START valid
    if not seen_start:
        valid[start_id] = True
        return valid

    # Case 4: At root level (depth=0), not closed -> OBJ_START/ARRAY_START
    if depth == 0:
        valid[obj_start_id] = True
        valid[array_start_id] = True
        return valid

    # Get current container type
    current_container = stack[depth - 1]

    # Case 5: Inside object, awaiting value -> values + containers
    if current_container == STACK_OBJECT and awaiting_value:
        for vid in value_ids:
            valid[vid] = True
        valid[obj_start_id] = True
        valid[array_start_id] = True
        return valid

    # Case 6: Inside object, not awaiting value -> keys + OBJ_END
    if current_container == STACK_OBJECT and not awaiting_value:
        for kid in key_ids:
            valid[kid] = True
        valid[obj_end_id] = True
        return valid

    # Case 7: Inside array -> values + containers + ARRAY_END
    if current_container == STACK_ARRAY:
        for vid in value_ids:
            valid[vid] = True
        valid[obj_start_id] = True
        valid[array_start_id] = True
        valid[array_end_id] = True
        return valid

    return valid


@njit(cache=True)
def _compute_single_sequence_mask(
    token_ids: np.ndarray,  # (seq_len,)
    vocab_size: int,
    max_depth: int,
    # Token IDs
    start_id: int,
    end_id: int,
    obj_start_id: int,
    obj_end_id: int,
    array_start_id: int,
    array_end_id: int,
    pad_id: int,
    key_ids: np.ndarray,
    value_ids: np.ndarray,
) -> np.ndarray:
    """Compute grammar mask for a single sequence.

    Returns: (seq_len, vocab_size) boolean mask
    """
    seq_len = len(token_ids)
    masks = np.zeros((seq_len, vocab_size), dtype=np.bool_)

    # Initialize state
    stack = np.zeros(max_depth, dtype=np.int64)
    depth = 0
    awaiting_value = 0
    seen_start = 0
    root_closed = 0
    ended = 0

    # Process each position
    for t in range(seq_len):
        token = token_ids[t]

        # Update state with current token
        depth, awaiting_value, seen_start, root_closed, ended = _update_state(
            token,
            stack,
            depth,
            awaiting_value,
            seen_start,
            root_closed,
            ended,
            start_id,
            end_id,
            obj_start_id,
            obj_end_id,
            array_start_id,
            array_end_id,
            pad_id,
            key_ids,
            value_ids,
            max_depth,
        )

        # Get valid mask for next position
        masks[t] = _get_valid_mask(
            stack,
            depth,
            awaiting_value,
            seen_start,
            root_closed,
            ended,
            start_id,
            end_id,
            obj_start_id,
            obj_end_id,
            array_start_id,
            array_end_id,
            pad_id,
            key_ids,
            value_ids,
            vocab_size,
        )

    return masks


@njit(parallel=True, cache=True)
def compute_grammar_mask_parallel(
    token_ids: np.ndarray,  # (batch_size, seq_len)
    vocab_size: int,
    max_depth: int,
    # Token IDs
    start_id: int,
    end_id: int,
    obj_start_id: int,
    obj_end_id: int,
    array_start_id: int,
    array_end_id: int,
    pad_id: int,
    key_ids: np.ndarray,
    value_ids: np.ndarray,
) -> np.ndarray:
    """Compute grammar masks for a batch in parallel.

    Uses Numba's prange to parallelize across sequences in the batch.
    Each sequence is processed independently, with positions processed
    sequentially within each sequence.

    Args:
        token_ids: (batch_size, seq_len) array of token IDs
        vocab_size: Size of vocabulary
        max_depth: Maximum nesting depth
        start_id, end_id, etc.: Special token IDs
        key_ids: Array of key token IDs
        value_ids: Array of value token IDs

    Returns:
        (batch_size, seq_len, vocab_size) boolean mask
    """
    batch_size, seq_len = token_ids.shape
    masks = np.zeros((batch_size, seq_len, vocab_size), dtype=np.bool_)

    # Parallel loop across sequences
    for b in prange(batch_size):
        masks[b] = _compute_single_sequence_mask(
            token_ids[b],
            vocab_size,
            max_depth,
            start_id,
            end_id,
            obj_start_id,
            obj_end_id,
            array_start_id,
            array_end_id,
            pad_id,
            key_ids,
            value_ids,
        )

    return masks
