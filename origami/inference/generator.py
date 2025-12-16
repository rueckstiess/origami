"""ORIGAMI JSON generator.

Generates complete JSON objects by autoregressive sampling from a trained ORIGAMI model.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from origami.position_encoding import PATH_TYPE_INDEX, PATH_TYPE_KEY
from origami.tokenizer.vocabulary import (
    ARRAY_END,
    ARRAY_START,
    END,
    OBJ_END,
    OBJ_START,
    START,
    KeyToken,
    ValueToken,
)

if TYPE_CHECKING:
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import JSONTokenizer


@dataclass
class PathState:
    """Tracks path state for a single sequence during generation.

    The path represents the current position in the JSON hierarchy.
    This mirrors how the tokenizer assigns paths during encoding.
    """

    # Stack of (context_type, path_elements) where context_type is 'object' or 'array'
    # path_elements is tuple of (type, id) pairs
    context_stack: list[tuple[str, list[tuple[int, int]]]] = field(
        default_factory=list
    )

    # Current key for object context (set after seeing KeyToken)
    current_key: tuple[int, int] | None = None

    # Current array index for array context
    array_index: int = 0

    def get_current_path(self) -> list[tuple[int, int]]:
        """Get the path for the current position."""
        if not self.context_stack:
            return []
        return list(self.context_stack[-1][1])

    def push_object(self) -> None:
        """Push a new object context onto the stack.

        If there's a current_key set, the new context's base path includes it.
        """
        # Use value path to include current key if present
        base_path = self.get_value_path()
        self.context_stack.append(("object", base_path))
        self.current_key = None  # Clear after consuming

    def push_array(self) -> None:
        """Push a new array context onto the stack.

        If there's a current_key set, the new context's base path includes it.
        """
        # Use value path to include current key if present
        base_path = self.get_value_path()
        self.context_stack.append(("array", base_path))
        self.current_key = None  # Clear after consuming
        self.array_index = 0

    def pop_context(self) -> None:
        """Pop the current context from the stack."""
        if self.context_stack:
            self.context_stack.pop()
        self.current_key = None

    def set_key(self, key_type: int, key_id: int) -> None:
        """Set the current key for an object context."""
        self.current_key = (key_type, key_id)

    def get_value_path(self) -> list[tuple[int, int]]:
        """Get the path for a value token (includes the key/index)."""
        if not self.context_stack:
            return []

        context_type, base_path = self.context_stack[-1]
        path = list(base_path)

        if context_type == "object" and self.current_key is not None:
            path.append(self.current_key)
        elif context_type == "array":
            path.append((PATH_TYPE_INDEX, self.array_index))

        return path

    def advance_array_index(self) -> None:
        """Advance the array index after processing an element."""
        self.array_index += 1

    def clone(self) -> "PathState":
        """Create a deep copy of the path state."""
        new_state = PathState()
        new_state.context_stack = [
            (ctx_type, list(path)) for ctx_type, path in self.context_stack
        ]
        new_state.current_key = self.current_key
        new_state.array_index = self.array_index
        return new_state


class OrigamiGenerator:
    """Generate JSON objects by autoregressive sampling from a trained model.

    Supports various sampling strategies (greedy, temperature, top-k, top-p)
    and can generate from scratch or continue from a partial object.

    Example:
        ```python
        generator = OrigamiGenerator(model, tokenizer)

        # Generate from scratch
        objects = generator.generate(num_samples=5, temperature=0.8)

        # Continue from partial object
        prefix = {"name": "Alice", "age": 30}
        completions = generator.generate_from_prefix(prefix, num_samples=3)
        ```
    """

    def __init__(
        self,
        model: "OrigamiModel",
        tokenizer: "JSONTokenizer",
    ):
        """Initialize generator.

        Args:
            model: Trained ORIGAMI model
            tokenizer: JSONTokenizer with fitted vocabulary

        Note:
            Generator always runs on CPU as benchmarking shows it's faster
            for the model sizes typically used with ORIGAMI.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        num_samples: int = 1,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """Generate complete JSON objects from scratch.

        Starts from [START] token and generates until [END] is produced.

        Args:
            num_samples: Number of objects to generate
            max_length: Maximum sequence length (default: 512)
            temperature: Sampling temperature (1.0 = unchanged, <1.0 = more greedy)
            top_k: If set, only sample from top-k most likely tokens
            top_p: If set, sample from smallest set with cumulative prob >= top_p
            seed: Random seed for reproducibility

        Returns:
            List of generated JSON objects
        """
        if seed is not None:
            torch.manual_seed(seed)

        max_length = max_length or 512
        vocab = self.tokenizer.vocab

        # Initialize with START token
        input_ids = torch.full(
            (num_samples, 1), vocab.start_id, dtype=torch.long, device=self.device
        )

        # Initialize path tensors
        path_types = torch.zeros(
            num_samples, 1, self.tokenizer.max_depth, dtype=torch.long, device=self.device
        )
        path_ids = torch.zeros(
            num_samples, 1, self.tokenizer.max_depth, dtype=torch.long, device=self.device
        )
        path_lengths = torch.zeros(
            num_samples, 1, dtype=torch.long, device=self.device
        )

        # Track path state for each sequence
        path_states = [PathState() for _ in range(num_samples)]

        # Track completion
        done = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        # Generate tokens
        for _ in range(max_length - 1):
            if done.all():
                break

            # Forward pass
            attention_mask = torch.ones(
                input_ids.shape, dtype=torch.bool, device=self.device
            )
            output = self.model(
                input_ids=input_ids,
                path_types=path_types,
                path_ids=path_ids,
                path_lengths=path_lengths,
                attention_mask=attention_mask,
            )

            # Get logits for last position
            next_logits = output.logits[:, -1, :]  # (batch, vocab_size)

            # Sample next token
            next_tokens = self._sample(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # For completed sequences, use PAD token
            next_tokens = torch.where(done, vocab.pad_token_id, next_tokens)

            # Check for END token
            done = done | (next_tokens == vocab.end_id)

            # Update path states and get new path tensors
            new_path_types, new_path_ids, new_path_lengths = self._update_paths(
                next_tokens, path_states, done
            )

            # Append new tokens and paths
            input_ids = torch.cat(
                [input_ids, next_tokens.unsqueeze(1)], dim=1
            )
            path_types = torch.cat([path_types, new_path_types], dim=1)
            path_ids = torch.cat([path_ids, new_path_ids], dim=1)
            path_lengths = torch.cat([path_lengths, new_path_lengths], dim=1)

        # Decode generated sequences
        results = []
        for i in range(num_samples):
            # Find END token position
            seq = input_ids[i].tolist()
            try:
                end_pos = seq.index(vocab.end_id)
                seq = seq[: end_pos + 1]
            except ValueError:
                # No END token, append it
                seq.append(vocab.end_id)

            try:
                obj = self.tokenizer.decode(seq)
                results.append(obj)
            except Exception:
                # Decoding failed, return empty dict
                results.append({})

        return results

    @torch.inference_mode()
    def generate_from_prefix(
        self,
        prefix: dict,
        num_samples: int = 1,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[dict]:
        """Continue generation from a partial JSON object.

        Args:
            prefix: Partial object to continue from
            num_samples: Number of completions to generate
            max_length: Maximum total sequence length (default: 512)
            temperature: Sampling temperature
            top_k: If set, only sample from top-k most likely tokens
            top_p: If set, sample from smallest set with cumulative prob >= top_p

        Returns:
            List of completed JSON objects
        """
        max_length = max_length or 512

        # Tokenize prefix (without END token)
        prefix_batch = self.tokenizer.encode_batch([prefix], shuffle=False)

        # Remove END token from prefix
        # The tokenizer adds START ... END, we want START ...
        prefix_len = prefix_batch.lengths[0].item() - 1  # Exclude END

        # Replicate prefix for all samples
        input_ids = prefix_batch.input_ids[:, :prefix_len].repeat(num_samples, 1)
        path_types = prefix_batch.path_types[:, :prefix_len].repeat(num_samples, 1, 1)
        path_ids = prefix_batch.path_ids[:, :prefix_len].repeat(num_samples, 1, 1)
        path_lengths = prefix_batch.path_lengths[:, :prefix_len].repeat(num_samples, 1)

        input_ids = input_ids.to(self.device)
        path_types = path_types.to(self.device)
        path_ids = path_ids.to(self.device)
        path_lengths = path_lengths.to(self.device)

        # Initialize path states from prefix
        path_states = self._init_path_states_from_tokens(
            input_ids[0].tolist(), num_samples
        )

        vocab = self.tokenizer.vocab
        done = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        # Generate tokens
        for _ in range(max_length - prefix_len):
            if done.all():
                break

            attention_mask = torch.ones(
                input_ids.shape, dtype=torch.bool, device=self.device
            )
            output = self.model(
                input_ids=input_ids,
                path_types=path_types,
                path_ids=path_ids,
                path_lengths=path_lengths,
                attention_mask=attention_mask,
            )

            next_logits = output.logits[:, -1, :]
            next_tokens = self._sample(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            next_tokens = torch.where(done, vocab.pad_token_id, next_tokens)
            done = done | (next_tokens == vocab.end_id)

            new_path_types, new_path_ids, new_path_lengths = self._update_paths(
                next_tokens, path_states, done
            )

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            path_types = torch.cat([path_types, new_path_types], dim=1)
            path_ids = torch.cat([path_ids, new_path_ids], dim=1)
            path_lengths = torch.cat([path_lengths, new_path_lengths], dim=1)

        # Decode generated sequences
        results = []
        for i in range(num_samples):
            seq = input_ids[i].tolist()
            try:
                end_pos = seq.index(vocab.end_id)
                seq = seq[: end_pos + 1]
            except ValueError:
                seq.append(vocab.end_id)

            try:
                obj = self.tokenizer.decode(seq)
                results.append(obj)
            except Exception:
                # Decoding failed, return empty dict
                results.append({})

        return results

    def generate_value(
        self,
        input_ids: Tensor,
        path_types: Tensor,
        path_ids: Tensor,
        path_lengths: Tensor,
        path_state: PathState,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> tuple[list[int], Any]:
        """Generate a complete value starting from current position.

        This is used by the Predictor to generate complex values (objects/arrays)
        when the predicted token is OBJ_START or ARRAY_START.

        Args:
            input_ids: Current token sequence (1, seq_len)
            path_types: Current path types (1, seq_len, max_depth)
            path_ids: Current path IDs (1, seq_len, max_depth)
            path_lengths: Current path lengths (1, seq_len)
            path_state: Current path state
            max_tokens: Maximum tokens to generate for the value
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering

        Returns:
            Tuple of (generated_token_ids, decoded_value)
        """
        from origami.constraints.json_grammar import JSONGrammarPDA

        vocab = self.tokenizer.vocab
        generated_tokens: list[int] = []

        # Clone tensors for generation
        current_ids = input_ids.clone()
        current_path_types = path_types.clone()
        current_path_ids = path_ids.clone()
        current_path_lengths = path_lengths.clone()
        current_state = path_state.clone()

        # Use PDA to track grammar state and determine when value is complete
        pda = JSONGrammarPDA(vocab, max_depth=self.tokenizer.max_depth)

        # Initialize PDA state by processing all tokens in input_ids
        # This gives us the state AFTER the OBJ_START/ARRAY_START
        batch_size = 1
        stack = torch.zeros(batch_size, pda.max_depth, dtype=torch.long, device=self.device)
        depth = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        awaiting_value = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        seen_start = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        root_closed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        ended = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Process existing tokens to get current PDA state
        for t in range(current_ids.size(1)):
            token = current_ids[:, t]
            stack, depth, awaiting_value, seen_start, root_closed, ended = pda._update_state(
                token, stack, depth, awaiting_value, seen_start, root_closed, ended
            )

        # Record initial depth (after OBJ_START/ARRAY_START)
        initial_depth = depth.item()

        for _ in range(max_tokens):
            attention_mask = torch.ones(
                current_ids.shape, dtype=torch.bool, device=self.device
            )
            output = self.model(
                input_ids=current_ids,
                path_types=current_path_types,
                path_ids=current_path_ids,
                path_lengths=current_path_lengths,
                attention_mask=attention_mask,
            )

            next_logits = output.logits[:, -1, :]
            next_token = self._sample(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            token_id = next_token.item()
            generated_tokens.append(token_id)

            # Update PDA state with generated token
            stack, depth, awaiting_value, seen_start, root_closed, ended = pda._update_state(
                next_token, stack, depth, awaiting_value, seen_start, root_closed, ended
            )

            # Value is complete when depth returns to initial_depth - 1
            # (we've closed the container that was opened by the start token)
            if depth.item() < initial_depth:
                break

            # Update path state for position encoding
            done = torch.zeros(1, dtype=torch.bool, device=self.device)
            new_path_types, new_path_ids, new_path_lengths = self._update_paths(
                next_token, [current_state], done
            )

            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)
            current_path_types = torch.cat([current_path_types, new_path_types], dim=1)
            current_path_ids = torch.cat([current_path_ids, new_path_ids], dim=1)
            current_path_lengths = torch.cat([current_path_lengths, new_path_lengths], dim=1)

        # Decode the generated value
        # We need to construct a minimal valid sequence to decode
        # The generated tokens form a complete value
        try:
            value = self._decode_value_tokens(generated_tokens)
        except Exception:
            value = None

        return generated_tokens, value

    def _decode_value_tokens(self, token_ids: list[int]) -> Any:
        """Decode a sequence of tokens representing a single value.

        Args:
            token_ids: Token IDs for a value (may be OBJ_START...OBJ_END or primitive)

        Returns:
            The decoded Python value
        """
        vocab = self.tokenizer.vocab

        if not token_ids:
            return None

        first_token = token_ids[0]

        if first_token == vocab.obj_start_id:
            # Parse object
            return self._parse_object_tokens(token_ids)
        elif first_token == vocab.array_start_id:
            # Parse array
            return self._parse_array_tokens(token_ids)
        else:
            # Primitive value
            token = vocab.decode(first_token)
            if isinstance(token, ValueToken):
                return token.value
            return None

    def _parse_object_tokens(self, token_ids: list[int]) -> dict:
        """Parse object tokens into a dictionary."""
        vocab = self.tokenizer.vocab
        result: dict[str, Any] = {}
        pos = 1  # Skip OBJ_START

        while pos < len(token_ids):
            token_id = token_ids[pos]

            if token_id == vocab.obj_end_id:
                break

            # Expect key
            token = vocab.decode(token_id)
            if not isinstance(token, KeyToken):
                break
            key = token.key
            pos += 1

            # Parse value
            value, pos = self._parse_value_at(token_ids, pos)
            result[key] = value

        return result

    def _parse_array_tokens(self, token_ids: list[int]) -> list:
        """Parse array tokens into a list."""
        vocab = self.tokenizer.vocab
        result: list[Any] = []
        pos = 1  # Skip ARRAY_START

        while pos < len(token_ids):
            token_id = token_ids[pos]

            if token_id == vocab.array_end_id:
                break

            value, pos = self._parse_value_at(token_ids, pos)
            result.append(value)

        return result

    def _parse_value_at(self, token_ids: list[int], pos: int) -> tuple[Any, int]:
        """Parse a value starting at position pos."""
        vocab = self.tokenizer.vocab

        if pos >= len(token_ids):
            return None, pos

        token_id = token_ids[pos]

        if token_id == vocab.obj_start_id:
            # Find matching OBJ_END
            depth = 1
            end_pos = pos + 1
            while end_pos < len(token_ids) and depth > 0:
                if token_ids[end_pos] == vocab.obj_start_id:
                    depth += 1
                elif token_ids[end_pos] == vocab.obj_end_id:
                    depth -= 1
                end_pos += 1
            obj = self._parse_object_tokens(token_ids[pos:end_pos])
            return obj, end_pos

        elif token_id == vocab.array_start_id:
            # Find matching ARRAY_END
            depth = 1
            end_pos = pos + 1
            while end_pos < len(token_ids) and depth > 0:
                if token_ids[end_pos] == vocab.array_start_id:
                    depth += 1
                elif token_ids[end_pos] == vocab.array_end_id:
                    depth -= 1
                end_pos += 1
            arr = self._parse_array_tokens(token_ids[pos:end_pos])
            return arr, end_pos

        else:
            # Primitive value
            token = vocab.decode(token_id)
            if isinstance(token, ValueToken):
                return token.value, pos + 1
            return None, pos + 1

    def _sample(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor:
        """Sample next tokens from logits.

        Args:
            logits: (batch, vocab_size) logits
            temperature: Temperature for scaling logits
            top_k: If set, only consider top-k tokens
            top_p: If set, use nucleus sampling

        Returns:
            Tensor of sampled token IDs, shape (batch,)
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_tokens

    def _update_paths(
        self,
        next_tokens: Tensor,
        path_states: list[PathState],
        done: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update path states and return new path tensors.

        Args:
            next_tokens: (batch,) next token IDs
            path_states: List of PathState for each sequence
            done: (batch,) boolean mask for completed sequences

        Returns:
            Tuple of (path_types, path_ids, path_lengths) for the new position
            Each has shape (batch, 1, max_depth) or (batch, 1)
        """
        vocab = self.tokenizer.vocab
        batch_size = len(path_states)
        max_depth = self.tokenizer.max_depth

        # Initialize output tensors
        new_path_types = torch.zeros(
            batch_size, 1, max_depth, dtype=torch.long, device=self.device
        )
        new_path_ids = torch.zeros(
            batch_size, 1, max_depth, dtype=torch.long, device=self.device
        )
        new_path_lengths = torch.zeros(
            batch_size, 1, dtype=torch.long, device=self.device
        )

        for i, (token_id, state, is_done) in enumerate(
            zip(next_tokens.tolist(), path_states, done.tolist())
        ):
            if is_done:
                continue

            # Determine path for this token based on token type
            token = vocab.decode(token_id)

            if token_id == vocab.start_id or token_id == vocab.end_id:
                # START and END have empty path
                path = []

            elif token_id == vocab.obj_start_id:
                # OBJ_START: when used as a value, path includes the key/index
                # get_value_path must be called BEFORE push_object clears current_key
                path = state.get_value_path()
                state.push_object()

            elif token_id == vocab.obj_end_id:
                # OBJ_END: path is the context's base path (same as its OBJ_START)
                path = state.get_current_path()
                state.pop_context()

            elif token_id == vocab.array_start_id:
                # ARRAY_START: when used as a value, path includes the key/index
                path = state.get_value_path()
                state.push_array()

            elif token_id == vocab.array_end_id:
                # ARRAY_END: path is the context's base path (same as its ARRAY_START)
                path = state.get_current_path()
                state.pop_context()

            elif isinstance(token, KeyToken):
                # Key token: path is containing object's path
                path = state.get_current_path()
                # Set current key for the upcoming value
                key_id = vocab.encode(token)
                state.set_key(PATH_TYPE_KEY, key_id)

            elif isinstance(token, ValueToken) or token_id == vocab.num_token_id:
                # Value token: path includes the key/index
                path = state.get_value_path()
                # Clear current key and advance array index if in array
                if state.context_stack and state.context_stack[-1][0] == "array":
                    state.advance_array_index()
                state.current_key = None

            else:
                # Unknown token type, use current path
                path = state.get_current_path()

            # Fill path tensors
            depth = min(len(path), max_depth)
            new_path_lengths[i, 0] = depth
            for d, (ptype, pid) in enumerate(path[:depth]):
                new_path_types[i, 0, d] = ptype
                new_path_ids[i, 0, d] = pid

        return new_path_types, new_path_ids, new_path_lengths

    def _init_path_states_from_tokens(
        self, token_ids: list[int], num_samples: int
    ) -> list[PathState]:
        """Initialize path states by replaying token sequence.

        Args:
            token_ids: Token sequence to replay
            num_samples: Number of copies to create

        Returns:
            List of PathState instances
        """
        vocab = self.tokenizer.vocab
        state = PathState()

        for token_id in token_ids:
            token = vocab.decode(token_id)

            if token_id == vocab.obj_start_id:
                state.push_object()
            elif token_id == vocab.obj_end_id:
                state.pop_context()
            elif token_id == vocab.array_start_id:
                state.push_array()
            elif token_id == vocab.array_end_id:
                state.pop_context()
            elif isinstance(token, KeyToken):
                key_id = vocab.encode(token)
                state.set_key(PATH_TYPE_KEY, key_id)
            elif isinstance(token, ValueToken) or token_id == vocab.num_token_id:
                if state.context_stack and state.context_stack[-1][0] == "array":
                    state.advance_array_index()
                state.current_key = None

        # Create copies for each sample
        return [state.clone() for _ in range(num_samples)]
