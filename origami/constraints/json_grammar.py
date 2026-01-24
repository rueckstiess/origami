"""JSON Grammar Pushdown Automaton for ORIGAMI.

Implements batch-parallel grammar constraint computation using a
sequential-over-positions, parallel-over-batch approach.

The grammar mask computation is O(seq_len) per batch with vectorized operations
across the batch dimension. For maximum throughput during training, use multiple
DataLoader workers (dataloader_num_workers config) to prepare batches in parallel.

When Numba is available, an optimized parallel implementation is used that
processes sequences in parallel across CPU cores.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from origami.tokenizer.vocabulary import Vocabulary

from .json_grammar_numba import NUMBA_AVAILABLE, compute_grammar_mask_parallel

# Stack content types
STACK_EMPTY = 0
STACK_OBJECT = 1
STACK_ARRAY = 2


class JSONGrammarPDA:
    """Pushdown automaton for JSON grammar constraints.

    Computes valid next-token masks based on JSON grammar rules.
    Uses a batch-parallel approach: sequential over positions,
    vectorized over batch dimension.

    The grammar rules enforced:
    - After START: OBJ_START or ARRAY_START (root container)
    - After OBJ_START: any key or OBJ_END
    - After key: value (primitive, NUM, OBJ_START, ARRAY_START)
    - After value in object: key or OBJ_END
    - After ARRAY_START: value or ARRAY_END
    - After value in array: value or ARRAY_END
    - After root closes: END only
    - After END: PAD only

    Attributes:
        vocab: Vocabulary instance for token type queries
        max_depth: Maximum nesting depth supported
    """

    def __init__(self, vocab: Vocabulary, max_depth: int = 32):
        """Initialize the grammar PDA.

        Args:
            vocab: Vocabulary instance
            max_depth: Maximum nesting depth (default 32)
        """
        self.vocab = vocab
        self.max_depth = max_depth

        # Pre-compute token ID tensors
        self._key_ids = torch.tensor(sorted(vocab.get_all_key_ids()), dtype=torch.long)
        self._value_ids = torch.tensor(
            sorted(vocab.get_all_primitive_value_ids()), dtype=torch.long
        )

        # Pre-compute mask patterns for vectorized operations
        vocab_size = vocab.size

        # Mask for all keys + OBJ_END (valid after value in object)
        self._keys_and_obj_end_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self._keys_and_obj_end_mask[self._key_ids] = True
        self._keys_and_obj_end_mask[vocab.obj_end_id] = True

        # Mask for all values + OBJ_START + ARRAY_START (valid after key in object)
        self._values_and_containers_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self._values_and_containers_mask[self._value_ids] = True
        self._values_and_containers_mask[vocab.obj_start_id] = True
        self._values_and_containers_mask[vocab.array_start_id] = True

        # Mask for array elements (values + containers + ARRAY_END)
        self._array_elements_mask = self._values_and_containers_mask.clone()
        self._array_elements_mask[vocab.array_end_id] = True

        # Cache for device-specific tensors (lazy initialization)
        self._device_masks: dict[torch.device, tuple[Tensor, Tensor, Tensor]] = {}
        self._device_ids: dict[torch.device, tuple[Tensor, Tensor]] = {}

        # NumPy arrays for Numba (if available)
        if NUMBA_AVAILABLE:
            self._key_ids_np = np.array(sorted(vocab.get_all_key_ids()), dtype=np.int64)
            self._value_ids_np = np.array(
                sorted(vocab.get_all_primitive_value_ids()), dtype=np.int64
            )

    def compute_valid_mask(
        self,
        token_ids: Tensor,  # (batch, seq_len)
        use_numba: bool = True,
    ) -> Tensor:
        """Compute grammar-valid next-token masks for each position.

        For autoregressive models: logits[t] predicts the token at position t+1.
        This method returns mask[t] indicating which tokens are valid at
        position t+1, based on state after processing tokens 0..t.

        When Numba is available and use_numba=True, uses an optimized parallel
        implementation that processes sequences in parallel across CPU cores.

        Note: With left-padded sequences, PAD tokens at the start are handled
        correctly by the grammar state (they don't update state). The loss
        computation handles PAD positions via attention_mask separately.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len)
            use_numba: If True and Numba available, use parallel implementation.

        Returns:
            Boolean mask of shape (batch, seq_len, vocab_size) where
            mask[t] indicates valid tokens for position t+1.
        """
        device = token_ids.device

        # Use Numba parallel implementation when available and on CPU
        if use_numba and NUMBA_AVAILABLE and device.type == "cpu":
            return self._compute_valid_mask_numba(token_ids)

        # Fallback to PyTorch implementation
        return self._compute_valid_mask_pytorch(token_ids)

    def _compute_valid_mask_numba(self, token_ids: Tensor) -> Tensor:
        """Numba-accelerated grammar mask computation.

        Converts to NumPy, runs parallel computation, converts back to tensor.
        """
        device = token_ids.device
        token_ids_np = token_ids.numpy()

        masks_np = compute_grammar_mask_parallel(
            token_ids_np,
            vocab_size=self.vocab.size,
            max_depth=self.max_depth,
            start_id=self.vocab.start_id,
            end_id=self.vocab.end_id,
            obj_start_id=self.vocab.obj_start_id,
            obj_end_id=self.vocab.obj_end_id,
            array_start_id=self.vocab.array_start_id,
            array_end_id=self.vocab.array_end_id,
            pad_id=self.vocab.pad_token_id,
            key_ids=self._key_ids_np,
            value_ids=self._value_ids_np,
        )

        return torch.from_numpy(masks_np).to(device)

    def _compute_valid_mask_pytorch(self, token_ids: Tensor) -> Tensor:
        """PyTorch implementation of grammar mask computation."""
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        vocab_size = self.vocab.size

        # Initialize output mask
        masks = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.bool, device=device)

        # Initialize state
        state = self.init_state(batch_size, device)

        # Process each position using the same code path as incremental generation
        for t in range(seq_len):
            current_token = token_ids[:, t]
            valid_mask, state = self.get_next_token_mask(current_token, state)
            masks[:, t] = valid_mask

        return masks

    def _get_valid_tokens(
        self,
        stack: Tensor,  # (batch, max_depth)
        depth: Tensor,  # (batch,)
        awaiting_value: Tensor,  # (batch,)
        seen_start: Tensor,  # (batch,)
        root_closed: Tensor,  # (batch,)
        ended: Tensor,  # (batch,)
        _key_ids: Tensor,  # (n_keys,) - unused, kept for API compatibility
        _value_ids: Tensor,  # (n_values,) - unused, kept for API compatibility
        device: torch.device,
    ) -> Tensor:
        """Get valid tokens at current position based on PDA state.

        Uses pre-computed mask patterns for O(1) vectorized operations
        instead of O(vocab_size) loops.

        State transitions:
        1. Initial (not seen_start): Only START valid
        2. After START, before root opens (seen_start, depth=0, not root_closed): OBJ_START/ARRAY_START
        3. Inside containers (depth > 0): Normal grammar rules
        4. After root closes (root_closed, not ended): Only END valid
        5. After END (ended): Only PAD valid

        Returns:
            Boolean mask of shape (batch, vocab_size)
        """
        batch_size = stack.shape[0]
        vocab_size = self.vocab.size

        # Initialize all-false mask
        valid = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)

        # Get current container type (0 if depth=0)
        depth_idx = (depth - 1).clamp(min=0).unsqueeze(1)  # (batch, 1)
        current_container = torch.gather(stack, 1, depth_idx).squeeze(1)  # (batch,)
        current_container = torch.where(
            depth == 0, torch.zeros_like(current_container), current_container
        )

        # Get cached device-specific masks (avoids repeated .to() calls)
        if device not in self._device_masks:
            self._device_masks[device] = (
                self._keys_and_obj_end_mask.to(device),
                self._values_and_containers_mask.to(device),
                self._array_elements_mask.to(device),
            )
        keys_and_obj_end, values_and_containers, array_elements = self._device_masks[device]

        # Case 1: After END -> only PAD valid
        valid[ended, self.vocab.pad_token_id] = True

        # Case 2: Root closed, not ended -> only END valid
        root_closed_not_ended = root_closed & ~ended
        valid[root_closed_not_ended, self.vocab.end_id] = True

        # Case 3: Not seen START yet -> only START valid
        not_started = ~seen_start & ~ended
        valid[not_started, self.vocab.start_id] = True

        # Case 4: Seen START, at root level (depth=0), root not closed -> OBJ_START/ARRAY_START
        ready_for_root = seen_start & (depth == 0) & ~root_closed & ~ended
        valid[ready_for_root, self.vocab.obj_start_id] = True
        valid[ready_for_root, self.vocab.array_start_id] = True

        # Case 5: Inside object, awaiting value -> values + containers (vectorized)
        in_obj_awaiting_val = (current_container == STACK_OBJECT) & awaiting_value & ~ended
        # Use outer product: (batch,) x (vocab,) -> (batch, vocab)
        valid = valid | (in_obj_awaiting_val.unsqueeze(1) & values_and_containers.unsqueeze(0))

        # Case 6: Inside object, not awaiting value -> keys + OBJ_END (vectorized)
        in_obj_not_awaiting = (current_container == STACK_OBJECT) & ~awaiting_value & ~ended
        valid = valid | (in_obj_not_awaiting.unsqueeze(1) & keys_and_obj_end.unsqueeze(0))

        # Case 7: Inside array -> values + containers + ARRAY_END (vectorized)
        in_array = (current_container == STACK_ARRAY) & ~ended
        valid = valid | (in_array.unsqueeze(1) & array_elements.unsqueeze(0))

        return valid

    def _update_state(
        self,
        token: Tensor,  # (batch,)
        stack: Tensor,  # (batch, max_depth)
        depth: Tensor,  # (batch,)
        awaiting_value: Tensor,  # (batch,)
        seen_start: Tensor,  # (batch,)
        root_closed: Tensor,  # (batch,)
        ended: Tensor,  # (batch,)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Update PDA state based on current token.

        Returns:
            Updated (stack, depth, awaiting_value, seen_start, root_closed, ended)
        """
        batch_size = token.shape[0]
        device = token.device

        # Note: All operations below (torch.where, |, scatter) return new tensors
        # rather than modifying in-place, so we can safely reassign the parameters.

        # Token type masks
        is_start = token == self.vocab.start_id
        is_end = token == self.vocab.end_id
        is_obj_start = token == self.vocab.obj_start_id
        is_obj_end = token == self.vocab.obj_end_id
        is_array_start = token == self.vocab.array_start_id
        is_array_end = token == self.vocab.array_end_id

        # Check if token is a key or value (vectorized with torch.isin)
        # Use cached device-specific tensors to avoid repeated .to() calls
        if device not in self._device_ids:
            self._device_ids[device] = (
                self._key_ids.to(device),
                self._value_ids.to(device),
            )
        key_ids, value_ids = self._device_ids[device]
        is_key = torch.isin(token, key_ids)
        is_value = torch.isin(token, value_ids)

        # START: mark seen_start = True
        seen_start = seen_start | is_start

        # END: mark as ended
        ended = ended | is_end

        # OBJ_START: push object onto stack
        push_obj = is_obj_start & (depth < self.max_depth)
        depth_indices = depth.unsqueeze(1)  # (batch, 1)
        obj_values = torch.full((batch_size, 1), STACK_OBJECT, dtype=torch.long, device=device)
        stack = torch.where(
            push_obj.unsqueeze(1).expand(-1, self.max_depth),
            stack.scatter(1, depth_indices.clamp(max=self.max_depth - 1), obj_values),
            stack,
        )
        depth = torch.where(push_obj, depth + 1, depth)
        awaiting_value = torch.where(push_obj, torch.zeros_like(awaiting_value), awaiting_value)

        # ARRAY_START: push array onto stack
        push_array = is_array_start & (depth < self.max_depth)
        array_values = torch.full((batch_size, 1), STACK_ARRAY, dtype=torch.long, device=device)
        stack = torch.where(
            push_array.unsqueeze(1).expand(-1, self.max_depth),
            stack.scatter(1, depth.unsqueeze(1).clamp(max=self.max_depth - 1), array_values),
            stack,
        )
        depth = torch.where(push_array, depth + 1, depth)

        # OBJ_END: pop from stack, check if closing root
        pop_obj = is_obj_end & (depth > 0)
        # Check if we're closing the root (depth will become 0)
        closing_root_obj = pop_obj & (depth == 1)
        depth = torch.where(pop_obj, depth - 1, depth)

        # ARRAY_END: pop from stack, check if closing root
        pop_array = is_array_end & (depth > 0)
        closing_root_array = pop_array & (depth == 1)
        depth = torch.where(pop_array, depth - 1, depth)

        # Mark root as closed
        root_closed = root_closed | closing_root_obj | closing_root_array

        # Key: set awaiting_value = True
        awaiting_value = torch.where(is_key, torch.ones_like(awaiting_value), awaiting_value)

        # Value (primitive): set awaiting_value = False
        awaiting_value = torch.where(is_value, torch.zeros_like(awaiting_value), awaiting_value)

        # After OBJ_END or ARRAY_END, if we're back in an object context,
        # the closed container was the value, so awaiting_value = False
        depth_idx = (depth - 1).clamp(min=0).unsqueeze(1)
        parent_container = torch.gather(stack, 1, depth_idx).squeeze(1)
        parent_is_obj = (parent_container == STACK_OBJECT) & (depth > 0)
        just_closed = pop_obj | pop_array
        awaiting_value = torch.where(
            just_closed & parent_is_obj,
            torch.zeros_like(awaiting_value),
            awaiting_value,
        )

        return stack, depth, awaiting_value, seen_start, root_closed, ended

    def init_state_from_tokens_batch(
        self,
        token_ids: Tensor,  # (batch, seq_len) left-padded sequences
        attention_mask: Tensor,  # (batch, seq_len) True for real tokens
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Initialize PDA state for a batch of sequences in parallel.

        This is much faster than calling init_state_from_tokens for each
        sequence individually, as it processes all sequences together.

        Args:
            token_ids: Batch of token IDs (batch, seq_len), left-padded
            attention_mask: Boolean mask (batch, seq_len), True for real tokens
            device: Device for output tensors

        Returns:
            Tuple of (stack, depth, awaiting_value, seen_start, root_closed, ended)
            each with shape (batch,) or (batch, max_depth) for stack.
        """
        batch_size, seq_len = token_ids.shape

        # Initialize batch state
        stack = torch.zeros(batch_size, self.max_depth, dtype=torch.long, device=device)
        depth = torch.zeros(batch_size, dtype=torch.long, device=device)
        awaiting_value = torch.zeros(batch_size, dtype=torch.bool, device=device)
        seen_start = torch.zeros(batch_size, dtype=torch.bool, device=device)
        root_closed = torch.zeros(batch_size, dtype=torch.bool, device=device)
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Process each position in parallel across the batch
        # For left-padded sequences, PAD tokens won't affect state
        # because we use torch.where with attention_mask
        for t in range(seq_len):
            tokens_at_t = token_ids[:, t]  # (batch,)
            mask_at_t = attention_mask[:, t]  # (batch,)

            # Skip if all tokens at this position are padding
            if not mask_at_t.any():
                continue

            # Compute new state for this position
            new_state = self._update_state(
                tokens_at_t, stack, depth, awaiting_value, seen_start, root_closed, ended
            )
            new_stack, new_depth, new_awaiting, new_seen, new_root_closed, new_ended = new_state

            # Only update state for non-padded positions
            mask_expanded = mask_at_t.unsqueeze(1).expand(-1, self.max_depth)
            stack = torch.where(mask_expanded, new_stack, stack)
            depth = torch.where(mask_at_t, new_depth, depth)
            awaiting_value = torch.where(mask_at_t, new_awaiting, awaiting_value)
            seen_start = torch.where(mask_at_t, new_seen, seen_start)
            root_closed = torch.where(mask_at_t, new_root_closed, root_closed)
            ended = torch.where(mask_at_t, new_ended, ended)

        return stack, depth, awaiting_value, seen_start, root_closed, ended

    def apply_constraints(
        self,
        logits: Tensor,  # (batch, seq_len, vocab_size)
        valid_mask: Tensor,  # (batch, seq_len, vocab_size)
        masked_value: float = float("-inf"),
    ) -> Tensor:
        """Apply grammar constraints by masking invalid tokens.

        Args:
            logits: Raw logits from model
            valid_mask: Boolean mask from compute_valid_mask()
            masked_value: Value to set for invalid tokens (default -inf)

        Returns:
            Logits with invalid tokens masked
        """
        return logits.masked_fill(~valid_mask, masked_value)

    def get_next_token_mask(
        self,
        last_token: Tensor,  # (batch,)
        state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Update state with one token and return next-token mask.

        For incremental generation where we maintain state between steps.
        This is O(1) per token instead of O(seq_len).

        The state tuple contains: (stack, depth, awaiting_value, seen_start, root_closed, ended)
        - stack: (batch, max_depth) - container type at each depth
        - depth: (batch,) - current nesting depth
        - awaiting_value: (batch,) - whether we're expecting a value after a key
        - seen_start: (batch,) - whether START token has been seen
        - root_closed: (batch,) - whether root container has been closed
        - ended: (batch,) - whether END token has been seen

        Args:
            last_token: The last generated token (batch,)
            state: Current PDA state tuple

        Returns:
            Tuple of (valid_mask, updated_state) where:
            - valid_mask: (batch, vocab_size) boolean mask for next valid tokens
            - updated_state: Updated state tuple after processing last_token
        """
        stack, depth, awaiting_value, seen_start, root_closed, ended = state
        device = last_token.device

        # Update state with the new token
        stack, depth, awaiting_value, seen_start, root_closed, ended = self._update_state(
            last_token, stack, depth, awaiting_value, seen_start, root_closed, ended
        )

        # Get valid tokens for next position
        key_ids = self._key_ids.to(device)
        value_ids = self._value_ids.to(device)

        valid_mask = self._get_valid_tokens(
            stack, depth, awaiting_value, seen_start, root_closed, ended, key_ids, value_ids, device
        )

        new_state = (stack, depth, awaiting_value, seen_start, root_closed, ended)
        return valid_mask, new_state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Initialize PDA state tensors for incremental generation.

        Returns:
            Tuple of (stack, depth, awaiting_value, seen_start, root_closed, ended)
        """
        stack = torch.zeros(batch_size, self.max_depth, dtype=torch.long, device=device)
        depth = torch.zeros(batch_size, dtype=torch.long, device=device)
        awaiting_value = torch.zeros(batch_size, dtype=torch.bool, device=device)
        seen_start = torch.zeros(batch_size, dtype=torch.bool, device=device)
        root_closed = torch.zeros(batch_size, dtype=torch.bool, device=device)
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return stack, depth, awaiting_value, seen_start, root_closed, ended

    def init_state_from_tokens(
        self,
        token_ids: Tensor,  # (seq_len,) single sequence
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Initialize PDA state by replaying a token sequence.

        Used when generating from a prefix - processes the prefix tokens
        to get the PDA state, then replicates that state for batch generation.

        Note: PAD tokens are skipped (for left-padded sequences).

        Args:
            token_ids: Single sequence of token IDs (seq_len,)
            batch_size: Number of copies to create for batched generation
            device: Device for output tensors

        Returns:
            Tuple of (stack, depth, awaiting_value, seen_start, root_closed, ended)
            each with batch dimension.
        """
        # Initialize single-sequence state
        stack = torch.zeros(1, self.max_depth, dtype=torch.long, device=device)
        depth = torch.zeros(1, dtype=torch.long, device=device)
        awaiting_value = torch.zeros(1, dtype=torch.bool, device=device)
        seen_start = torch.zeros(1, dtype=torch.bool, device=device)
        root_closed = torch.zeros(1, dtype=torch.bool, device=device)
        ended = torch.zeros(1, dtype=torch.bool, device=device)

        # Process each token (skip PAD tokens for left-padded sequences)
        for t in range(token_ids.size(0)):
            token = token_ids[t : t + 1].to(device)  # (1,)
            if token.item() == self.vocab.pad_token_id:
                continue
            stack, depth, awaiting_value, seen_start, root_closed, ended = self._update_state(
                token, stack, depth, awaiting_value, seen_start, root_closed, ended
            )

        # Replicate state for batch
        stack = stack.expand(batch_size, -1).contiguous()
        depth = depth.expand(batch_size).contiguous()
        awaiting_value = awaiting_value.expand(batch_size).contiguous()
        seen_start = seen_start.expand(batch_size).contiguous()
        root_closed = root_closed.expand(batch_size).contiguous()
        ended = ended.expand(batch_size).contiguous()

        return stack, depth, awaiting_value, seen_start, root_closed, ended
