"""JSON Grammar Pushdown Automaton for ORIGAMI.

Implements batch-parallel grammar constraint computation using a
sequential-over-positions, parallel-over-batch approach.
"""

import torch
from torch import Tensor

from origami.tokenizer.vocabulary import Vocabulary


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

        # Pre-compute token ID sets for efficient mask creation
        self._key_ids = torch.tensor(sorted(vocab.get_all_key_ids()), dtype=torch.long)
        self._value_ids = torch.tensor(
            sorted(vocab.get_all_primitive_value_ids()), dtype=torch.long
        )

    def compute_valid_mask(
        self,
        token_ids: Tensor,  # (batch, seq_len)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
    ) -> Tensor:
        """Compute grammar-valid next-token masks for each position.

        For autoregressive models: logits[t] predicts the token at position t+1.
        This method returns mask[t] indicating which tokens are valid at
        position t+1, based on state after processing tokens 0..t.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len)
            attention_mask: Boolean mask where True = valid position.
                If None, all positions are valid.

        Returns:
            Boolean mask of shape (batch, seq_len, vocab_size) where
            mask[t] indicates valid tokens for position t+1.
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        vocab_size = self.vocab.size

        # Initialize state tensors (batch-parallel)
        # Stack stores container type at each depth level
        stack = torch.zeros(
            batch_size, self.max_depth, dtype=torch.long, device=device
        )
        # Current depth (0 = at root level)
        depth = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Whether we're awaiting a value (after key in object)
        awaiting_value = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Whether START has been seen
        seen_start = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Whether root container has been closed
        root_closed = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Whether END has been seen
        ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Output mask: valid tokens for position t
        masks = torch.zeros(
            batch_size, seq_len, vocab_size, dtype=torch.bool, device=device
        )

        # Move pre-computed ID tensors to device
        key_ids = self._key_ids.to(device)
        value_ids = self._value_ids.to(device)

        # Process each position sequentially
        # For autoregressive training: logits[t] predicts token at position t+1
        # So mask[t] should indicate valid tokens for position t+1, given tokens 0..t
        for t in range(seq_len):
            # Get current token
            current_token = token_ids[:, t]  # (batch,)

            # Update state with current token FIRST
            stack, depth, awaiting_value, seen_start, root_closed, ended = self._update_state(
                current_token, stack, depth, awaiting_value, seen_start, root_closed, ended
            )

            # Compute valid tokens for NEXT position (t+1) based on state after 0..t
            valid_mask = self._get_valid_tokens(
                stack, depth, awaiting_value, seen_start, root_closed, ended,
                key_ids, value_ids, device
            )

            # Apply attention mask: padding positions get PAD-only mask
            if attention_mask is not None:
                pad_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                pad_mask[self.vocab.pad_token_id] = True
                # Where attention_mask is False (padding), use pad-only mask
                valid_mask = torch.where(
                    attention_mask[:, t : t + 1],
                    valid_mask,
                    pad_mask.unsqueeze(0).expand(batch_size, -1),
                )

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
        key_ids: Tensor,  # (n_keys,)
        value_ids: Tensor,  # (n_values,)
        device: torch.device,
    ) -> Tensor:
        """Get valid tokens at current position based on PDA state.

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
        current_container = torch.where(depth == 0, torch.zeros_like(current_container), current_container)

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

        # Case 5: Inside object, awaiting value -> any value type
        in_obj_awaiting_val = (current_container == STACK_OBJECT) & awaiting_value & ~ended
        if in_obj_awaiting_val.any():
            # Set all value IDs valid for those positions
            for vid in value_ids:
                valid[in_obj_awaiting_val, vid.item()] = True
            # Nested containers
            valid[in_obj_awaiting_val, self.vocab.obj_start_id] = True
            valid[in_obj_awaiting_val, self.vocab.array_start_id] = True

        # Case 6: Inside object, not awaiting value -> key or OBJ_END
        in_obj_not_awaiting = (current_container == STACK_OBJECT) & ~awaiting_value & ~ended
        if in_obj_not_awaiting.any():
            # Set all key IDs valid for those positions
            for kid in key_ids:
                valid[in_obj_not_awaiting, kid.item()] = True
            # Close object
            valid[in_obj_not_awaiting, self.vocab.obj_end_id] = True

        # Case 7: Inside array -> any value or ARRAY_END
        in_array = (current_container == STACK_ARRAY) & ~ended
        if in_array.any():
            # Set all value IDs valid for those positions
            for vid in value_ids:
                valid[in_array, vid.item()] = True
            # Nested containers
            valid[in_array, self.vocab.obj_start_id] = True
            valid[in_array, self.vocab.array_start_id] = True
            # Close array
            valid[in_array, self.vocab.array_end_id] = True

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

        # Make copies to avoid in-place modification issues
        new_stack = stack.clone()
        new_depth = depth.clone()
        new_awaiting_value = awaiting_value.clone()
        new_seen_start = seen_start.clone()
        new_root_closed = root_closed.clone()
        new_ended = ended.clone()

        # Token type masks
        is_start = token == self.vocab.start_id
        is_end = token == self.vocab.end_id
        is_obj_start = token == self.vocab.obj_start_id
        is_obj_end = token == self.vocab.obj_end_id
        is_array_start = token == self.vocab.array_start_id
        is_array_end = token == self.vocab.array_end_id

        # Check if token is a key (vectorized)
        is_key = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for kid in self._key_ids:
            is_key = is_key | (token == kid.item())

        # Check if token is a value (primitive, vectorized)
        is_value = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for vid in self._value_ids:
            is_value = is_value | (token == vid.item())

        # START: mark seen_start = True
        new_seen_start = new_seen_start | is_start

        # END: mark as ended
        new_ended = new_ended | is_end

        # OBJ_START: push object onto stack
        push_obj = is_obj_start & (new_depth < self.max_depth)
        depth_indices = new_depth.unsqueeze(1)  # (batch, 1)
        obj_values = torch.full((batch_size, 1), STACK_OBJECT, dtype=torch.long, device=device)
        new_stack = torch.where(
            push_obj.unsqueeze(1).expand(-1, self.max_depth),
            new_stack.scatter(1, depth_indices.clamp(max=self.max_depth - 1), obj_values),
            new_stack,
        )
        new_depth = torch.where(push_obj, new_depth + 1, new_depth)
        new_awaiting_value = torch.where(push_obj, torch.zeros_like(new_awaiting_value), new_awaiting_value)

        # ARRAY_START: push array onto stack
        push_array = is_array_start & (new_depth < self.max_depth)
        array_values = torch.full((batch_size, 1), STACK_ARRAY, dtype=torch.long, device=device)
        new_stack = torch.where(
            push_array.unsqueeze(1).expand(-1, self.max_depth),
            new_stack.scatter(1, new_depth.unsqueeze(1).clamp(max=self.max_depth - 1), array_values),
            new_stack,
        )
        new_depth = torch.where(push_array, new_depth + 1, new_depth)

        # OBJ_END: pop from stack, check if closing root
        pop_obj = is_obj_end & (new_depth > 0)
        # Check if we're closing the root (depth will become 0)
        closing_root_obj = pop_obj & (new_depth == 1)
        new_depth = torch.where(pop_obj, new_depth - 1, new_depth)

        # ARRAY_END: pop from stack, check if closing root
        pop_array = is_array_end & (new_depth > 0)
        closing_root_array = pop_array & (new_depth == 1)
        new_depth = torch.where(pop_array, new_depth - 1, new_depth)

        # Mark root as closed
        new_root_closed = new_root_closed | closing_root_obj | closing_root_array

        # Key: set awaiting_value = True
        new_awaiting_value = torch.where(is_key, torch.ones_like(new_awaiting_value), new_awaiting_value)

        # Value (primitive): set awaiting_value = False
        new_awaiting_value = torch.where(is_value, torch.zeros_like(new_awaiting_value), new_awaiting_value)

        # After OBJ_END or ARRAY_END, if we're back in an object context,
        # the closed container was the value, so awaiting_value = False
        depth_idx = (new_depth - 1).clamp(min=0).unsqueeze(1)
        parent_container = torch.gather(new_stack, 1, depth_idx).squeeze(1)
        parent_is_obj = (parent_container == STACK_OBJECT) & (new_depth > 0)
        just_closed = pop_obj | pop_array
        new_awaiting_value = torch.where(
            just_closed & parent_is_obj,
            torch.zeros_like(new_awaiting_value),
            new_awaiting_value,
        )

        return new_stack, new_depth, new_awaiting_value, new_seen_start, new_root_closed, new_ended

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
