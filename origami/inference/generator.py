"""ORIGAMI JSON generator.

Generates complete JSON objects by autoregressive sampling from a trained ORIGAMI model.

The Generator provides two public methods:
- `generate()`: Generate complete JSON objects from scratch
- `generate_from_tensors()`: Core generation loop from pre-encoded sequences

The Predictor uses `generate_from_tensors()` with `stop_after_value=True` for value prediction.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from origami.position_encoding import PATH_TYPE_INDEX, PATH_TYPE_KEY
from origami.tokenizer.vocabulary import (
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
    context_stack: list[tuple[str, list[tuple[int, int]]]] = field(default_factory=list)

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
        new_state.context_stack = [(ctx_type, list(path)) for ctx_type, path in self.context_stack]
        new_state.current_key = self.current_key
        new_state.array_index = self.array_index
        return new_state


class OrigamiGenerator:
    """Generate JSON objects by autoregressive sampling from a trained model.

    Supports various sampling strategies (greedy, temperature, top-k, top-p).

    Public methods:
    - `generate()`: Generate complete JSON objects from scratch
    - `generate_from_tensors()`: Core generation loop from pre-encoded tensor sequences

    Example:
        ```python
        generator = OrigamiGenerator(model, tokenizer)

        # Generate from scratch
        objects = generator.generate(num_samples=5, temperature=0.8)
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

        # Get grammar PDA reference from model for incremental constraint application
        self._grammar_pda = model._grammar_pda

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
        max_depth = self.tokenizer.max_depth

        # Initialize with START token
        input_ids = torch.full(
            (num_samples, 1), vocab.start_id, dtype=torch.long, device=self.device
        )

        # Initialize path tensors (START has empty path)
        path_types = torch.zeros(num_samples, 1, max_depth, dtype=torch.long, device=self.device)
        path_ids = torch.zeros(num_samples, 1, max_depth, dtype=torch.long, device=self.device)
        path_lengths = torch.zeros(num_samples, 1, dtype=torch.long, device=self.device)

        # Attention mask: all ones since no padding yet
        attention_mask = torch.ones(num_samples, 1, dtype=torch.bool, device=self.device)

        # Delegate to core generation loop
        return self.generate_from_tensors(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            stop_after_value=False,  # Generate until END
            max_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    @torch.inference_mode()
    def generate_from_tensors(
        self,
        input_ids: Tensor,
        path_types: Tensor,
        path_ids: Tensor,
        path_lengths: Tensor,
        attention_mask: Tensor,
        numeric_values: Tensor | None = None,
        stop_after_value: bool = False,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[Any]:
        """Core generation loop from pre-encoded sequences.

        This is the single implementation of the generation loop.
        All generation methods call this internally.

        Uses dynamic batch compaction: completed sequences are removed from the
        batch to avoid unnecessary computation on finished sequences.

        Args:
            input_ids: Pre-encoded sequences (batch, seq_len), may be left-padded
            path_types: Path type encoding (batch, seq_len, max_depth)
            path_ids: Path ID encoding (batch, seq_len, max_depth)
            path_lengths: Path lengths (batch, seq_len)
            attention_mask: True for real tokens, False for PAD
            numeric_values: Optional numeric values for scaled fields (batch, seq_len).
                           Required for conditioning on scaled numeric context (e.g., prediction).
            stop_after_value: If True, stop each sequence after one complete value
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering

        Returns:
            List of generated values (one per input sequence).
            Values can be dicts, lists, or primitives depending on what was generated.
        """
        vocab = self.tokenizer.vocab
        original_batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Clone tensors for generation (we'll extend them)
        current_ids = input_ids.clone()
        current_path_types = path_types.clone()
        current_path_ids = path_ids.clone()
        current_path_lengths = path_lengths.clone()
        current_attention_mask = attention_mask.clone()

        # Track numeric values for continuous head
        # Use provided numeric_values for conditioning, or zeros if not provided
        if numeric_values is not None:
            current_numeric_values = numeric_values.clone().to(self.device)
        else:
            current_numeric_values = torch.zeros(
                original_batch_size, seq_len, dtype=torch.float, device=self.device
            )
        # Track sampled numeric values for later decoding (list of lists)
        sampled_numeric_values: list[list[float]] = [[] for _ in range(original_batch_size)]

        # Initialize path states for each sequence from their token prefixes
        path_states = []
        for i in range(original_batch_size):
            mask = current_attention_mask[i]
            seq_tokens = current_ids[i][mask].tolist()
            # Create single path state from prefix
            states = self._init_path_states_from_tokens(seq_tokens, num_samples=1)
            path_states.append(states[0])

        # Initialize grammar state for each sequence from their prefixes
        grammar_state = None
        initial_depths = None
        if self._grammar_pda is not None:
            grammar_states = []
            for i in range(original_batch_size):
                mask = current_attention_mask[i]
                seq_tokens = current_ids[i][mask]
                state = self._grammar_pda.init_state_from_tokens(seq_tokens, 1, self.device)
                grammar_states.append(state)
            grammar_state = self._stack_grammar_states(grammar_states)
            # Record initial depths for stop_after_value
            initial_depths = grammar_state[1].clone()

        # Track where each sequence's generated content starts
        gen_start_positions = current_ids.size(1) * torch.ones(
            original_batch_size, dtype=torch.long, device=self.device
        )

        # Track original indices for each active sequence (for reordering at end)
        active_indices = list(range(original_batch_size))

        # Store completed results: maps original index -> (seq_tokens, numeric_values)
        completed_results: dict[int, tuple[list[int], list[float]]] = {}

        # Generation loop
        for _ in range(max_tokens):
            if len(active_indices) == 0:
                break

            batch_size = current_ids.size(0)

            # Forward pass with numeric values for continuous head
            output = self.model(
                input_ids=current_ids,
                path_types=current_path_types,
                path_ids=current_path_ids,
                path_lengths=current_path_lengths,
                attention_mask=current_attention_mask,
                numeric_values=current_numeric_values if self.model.continuous_head is not None else None,
            )

            # Get logits for last position
            next_logits = output.logits[:, -1, :]  # (batch, vocab_size)

            # Apply grammar constraints incrementally - O(1) per step
            if self._grammar_pda is not None and grammar_state is not None:
                last_token = current_ids[:, -1]
                valid_mask, grammar_state = self._grammar_pda.get_next_token_mask(
                    last_token, grammar_state
                )
                next_logits = next_logits.masked_fill(~valid_mask, float("-inf"))

            # Sample next token
            next_tokens = self._sample(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Sample numeric values for NUM tokens from continuous head
            new_numeric_values = torch.zeros(batch_size, 1, dtype=torch.float, device=self.device)
            is_num = next_tokens == vocab.num_token_id
            if is_num.any() and output.continuous_params is not None:
                weights, means, log_vars = output.continuous_params
                # Get params for last position only
                w = weights[:, -1, :]  # (batch, n_components)
                m = means[:, -1, :]
                lv = log_vars[:, -1, :]
                # Sample from MoG for all sequences (even if they didn't generate NUM)
                sampled = self.model.continuous_head.sample(
                    w.unsqueeze(1), m.unsqueeze(1), lv.unsqueeze(1)
                ).squeeze(1)  # (batch,)
                # Only use sampled value where NUM was generated
                new_numeric_values[:, 0] = torch.where(is_num, sampled, new_numeric_values[:, 0])

            # Track sampled values for decoding
            for i in range(batch_size):
                orig_idx = active_indices[i]
                if is_num[i]:
                    sampled_numeric_values[orig_idx].append(new_numeric_values[i, 0].item())

            # Check for completion
            if stop_after_value and initial_depths is not None:
                # Stop when depth returns below initial (value is complete)
                current_depths = grammar_state[1]  # depth is second element
                just_completed = current_depths < initial_depths
            else:
                # Stop on END token
                just_completed = next_tokens == vocab.end_id

            # Update path states and get new path tensors
            new_path_types, new_path_ids, new_path_lengths = self._update_paths(
                next_tokens, path_states, just_completed
            )

            # Extend tensors with new tokens
            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)
            current_path_types = torch.cat([current_path_types, new_path_types], dim=1)
            current_path_ids = torch.cat([current_path_ids, new_path_ids], dim=1)
            current_path_lengths = torch.cat([current_path_lengths, new_path_lengths], dim=1)
            current_numeric_values = torch.cat([current_numeric_values, new_numeric_values], dim=1)
            new_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
            current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)

            # Store completed sequences and remove them from active batch
            if just_completed.any():
                completed_mask = just_completed.tolist()
                keep_indices = []
                new_active_indices = []
                new_path_states = []

                for i, (is_complete, orig_idx) in enumerate(
                    zip(completed_mask, active_indices, strict=True)
                ):
                    if is_complete:
                        # Store completed sequence
                        mask = current_attention_mask[i]
                        seq_tokens = current_ids[i][mask].tolist()
                        completed_results[orig_idx] = (
                            seq_tokens,
                            sampled_numeric_values[orig_idx],
                        )
                    else:
                        keep_indices.append(i)
                        new_active_indices.append(orig_idx)
                        new_path_states.append(path_states[i])

                # Compact tensors to only keep active sequences
                if keep_indices:
                    keep_tensor = torch.tensor(keep_indices, device=self.device)
                    current_ids = current_ids[keep_tensor]
                    current_path_types = current_path_types[keep_tensor]
                    current_path_ids = current_path_ids[keep_tensor]
                    current_path_lengths = current_path_lengths[keep_tensor]
                    current_numeric_values = current_numeric_values[keep_tensor]
                    current_attention_mask = current_attention_mask[keep_tensor]

                    # Compact grammar state
                    if grammar_state is not None:
                        grammar_state = tuple(s[keep_tensor] for s in grammar_state)
                        if initial_depths is not None:
                            initial_depths = initial_depths[keep_tensor]

                active_indices = new_active_indices
                path_states = new_path_states

        # Store any remaining sequences that didn't complete
        for i, orig_idx in enumerate(active_indices):
            mask = current_attention_mask[i]
            seq_tokens = current_ids[i][mask].tolist()
            completed_results[orig_idx] = (seq_tokens, sampled_numeric_values[orig_idx])

        # Decode all sequences in original order
        results = []
        for orig_idx in range(original_batch_size):
            seq, numeric_vals = completed_results[orig_idx]
            start_pos = gen_start_positions[orig_idx].item()

            if stop_after_value:
                # Decode just the generated value tokens
                value_tokens = seq[start_pos:]
                try:
                    value = self._decode_value_tokens(value_tokens, numeric_vals)
                    results.append(value)
                except Exception:
                    results.append(None)
            else:
                # Decode full sequence as JSON object
                # Find END token position
                try:
                    end_pos = seq.index(vocab.end_id)
                    seq = seq[: end_pos + 1]
                except ValueError:
                    # No END token, append it
                    seq.append(vocab.end_id)

                try:
                    # For full object decoding with NUM tokens, use our parser
                    obj = self._decode_with_numerics(seq, numeric_vals)
                    results.append(obj)
                except Exception:
                    results.append({})

        return results

    @torch.inference_mode()
    def get_value_distribution(
        self,
        input_ids: Tensor,
        path_types: Tensor,
        path_ids: Tensor,
        path_lengths: Tensor,
        attention_mask: Tensor,
        numeric_values: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        """Get grammar-constrained probability distribution for next token.

        This is used by Predictor.predict_proba() to get the distribution over
        possible values without actually sampling/generating.

        Args:
            input_ids: (batch, seq_len) token IDs ending at target key
            path_types: (batch, seq_len, max_depth) path type encoding
            path_ids: (batch, seq_len, max_depth) path element IDs
            path_lengths: (batch, seq_len) path lengths
            attention_mask: (batch, seq_len) attention mask
            numeric_values: Optional numeric values for conditioning (batch, seq_len)

        Returns:
            probs: (batch, vocab_size) probabilities after grammar masking
            continuous_params: Optional (weights, means, log_vars) for MoG head,
                               each with shape (batch, n_components)
        """
        # 1. Forward pass
        output = self.model(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            numeric_values=numeric_values,
        )

        # 2. Get logits at last position (predicts next token)
        next_logits = output.logits[:, -1, :]  # (batch, vocab_size)

        # 3. Apply grammar constraints
        if self._grammar_pda is not None:
            # Initialize grammar state from the full sequence
            batch_size = input_ids.size(0)
            grammar_states = []
            for i in range(batch_size):
                mask = attention_mask[i]
                seq_tokens = input_ids[i][mask]
                state = self._grammar_pda.init_state_from_tokens(seq_tokens, 1, self.device)
                grammar_states.append(state)
            grammar_state = self._stack_grammar_states(grammar_states)

            # Get valid next tokens based on grammar state
            last_token = input_ids[:, -1]
            valid_mask, _ = self._grammar_pda.get_next_token_mask(last_token, grammar_state)
            next_logits = next_logits.masked_fill(~valid_mask, float("-inf"))

        # 4. Convert to probabilities
        probs = F.softmax(next_logits, dim=-1)

        # 5. Get continuous params if available
        continuous_params = None
        if output.continuous_params is not None:
            weights, means, log_vars = output.continuous_params
            continuous_params = (
                weights[:, -1, :],  # (batch, n_components)
                means[:, -1, :],  # (batch, n_components)
                log_vars[:, -1, :],  # (batch, n_components)
            )

        return probs, continuous_params

    def _stack_grammar_states(
        self,
        states: list[tuple[Tensor, ...]],
    ) -> tuple[Tensor, ...]:
        """Stack individual grammar states into a batched state.

        Args:
            states: List of state tuples, each from init_state_from_tokens
                   with batch_size=1

        Returns:
            Single state tuple with concatenated batch dimension
        """
        # Each state is (stack, depth, awaiting_value, seen_start, root_closed, ended)
        # Each tensor has shape (1, ...) - we concatenate along batch dimension
        num_components = len(states[0])
        stacked = []
        for i in range(num_components):
            component_tensors = [s[i] for s in states]
            stacked.append(torch.cat(component_tensors, dim=0))
        return tuple(stacked)

    def _decode_value_tokens(
        self,
        token_ids: list[int],
        numeric_values: list[float] | None = None,
    ) -> Any:
        """Decode a sequence of tokens representing a single value.

        Args:
            token_ids: Token IDs for a value (may be OBJ_START...OBJ_END or primitive)
            numeric_values: List of sampled numeric values for NUM tokens

        Returns:
            The decoded Python value
        """
        vocab = self.tokenizer.vocab

        if not token_ids:
            return None

        first_token = token_ids[0]

        if first_token == vocab.obj_start_id:
            # Parse object
            num_idx = [0]  # Mutable counter for tracking NUM position
            return self._parse_object_tokens(token_ids, numeric_values, num_idx)
        elif first_token == vocab.array_start_id:
            # Parse array
            num_idx = [0]
            return self._parse_array_tokens(token_ids, numeric_values, num_idx)
        elif first_token == vocab.num_token_id:
            # NUM token - use sampled value
            if numeric_values:
                return numeric_values[0]
            return 0.0  # Fallback
        else:
            # Primitive value
            token = vocab.decode(first_token)
            if isinstance(token, ValueToken):
                return token.value
            return None

    def _decode_with_numerics(
        self,
        token_ids: list[int],
        numeric_values: list[float],
    ) -> dict:
        """Decode a full sequence with NUM token support.

        Args:
            token_ids: Full token sequence (START...END)
            numeric_values: List of sampled numeric values for NUM tokens

        Returns:
            Decoded JSON object
        """
        vocab = self.tokenizer.vocab

        if not token_ids:
            return {}

        # Skip START token
        pos = 0
        if token_ids[pos] == vocab.start_id:
            pos += 1

        # Expect OBJ_START
        if pos >= len(token_ids) or token_ids[pos] != vocab.obj_start_id:
            return {}

        # Parse object with numeric values
        num_idx = [0]  # Mutable counter
        return self._parse_object_tokens(token_ids[pos:], numeric_values, num_idx)

    def _parse_object_tokens(
        self,
        token_ids: list[int],
        numeric_values: list[float] | None = None,
        num_idx: list[int] | None = None,
    ) -> dict:
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
            value, pos = self._parse_value_at(token_ids, pos, numeric_values, num_idx)
            result[key] = value

        return result

    def _parse_array_tokens(
        self,
        token_ids: list[int],
        numeric_values: list[float] | None = None,
        num_idx: list[int] | None = None,
    ) -> list:
        """Parse array tokens into a list."""
        vocab = self.tokenizer.vocab
        result: list[Any] = []
        pos = 1  # Skip ARRAY_START

        while pos < len(token_ids):
            token_id = token_ids[pos]

            if token_id == vocab.array_end_id:
                break

            value, pos = self._parse_value_at(token_ids, pos, numeric_values, num_idx)
            result.append(value)

        return result

    def _parse_value_at(
        self,
        token_ids: list[int],
        pos: int,
        numeric_values: list[float] | None = None,
        num_idx: list[int] | None = None,
    ) -> tuple[Any, int]:
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
            obj = self._parse_object_tokens(token_ids[pos:end_pos], numeric_values, num_idx)
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
            arr = self._parse_array_tokens(token_ids[pos:end_pos], numeric_values, num_idx)
            return arr, end_pos

        elif token_id == vocab.num_token_id:
            # NUM token - use sampled value
            value = 0.0  # Default fallback
            if numeric_values and num_idx is not None and num_idx[0] < len(numeric_values):
                value = numeric_values[num_idx[0]]
                num_idx[0] += 1
            return value, pos + 1

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
        # Handle greedy decoding (temperature=0) with argmax
        if temperature == 0.0:
            return logits.argmax(dim=-1)

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
        new_path_types = torch.zeros(batch_size, 1, max_depth, dtype=torch.long, device=self.device)
        new_path_ids = torch.zeros(batch_size, 1, max_depth, dtype=torch.long, device=self.device)
        new_path_lengths = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

        for i, (token_id, state, is_done) in enumerate(
            zip(next_tokens.tolist(), path_states, done.tolist(), strict=True)
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
