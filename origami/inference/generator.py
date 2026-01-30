"""ORIGAMI JSON generator.

Generates complete JSON objects by autoregressive sampling from a trained ORIGAMI model.

The Generator is the SINGLE implementation of generation logic. All generation
(from scratch or from prefix) goes through generate_from_batch().

Public methods:
- `generate()`: Generate complete JSON objects from scratch
- `generate_from_batch()`: Core generation loop from EncodedBatch prefix
- `get_next_token_distribution()`: Get grammar-constrained probabilities (no sampling)
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

from .utils import GenerationError

if TYPE_CHECKING:
    from origami.constraints.schema_pda import SchemaState
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import EncodedBatch, JSONTokenizer


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

    # Track seen key token IDs at each object depth level
    # Each entry corresponds to an object in context_stack (only for objects, not arrays)
    seen_keys_stack: list[set[int]] = field(default_factory=list)

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
        self.seen_keys_stack.append(set())  # Track keys for this object
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
            ctx_type, _ = self.context_stack.pop()
            # Only objects have seen_keys tracking
            if ctx_type == "object" and self.seen_keys_stack:
                self.seen_keys_stack.pop()
        self.current_key = None

    def set_key(self, key_type: int, key_id: int) -> None:
        """Set the current key for an object context."""
        self.current_key = (key_type, key_id)
        # Record this key as seen in current object
        if self.seen_keys_stack:
            self.seen_keys_stack[-1].add(key_id)

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

    def get_seen_keys(self) -> set[int]:
        """Get the set of key token IDs seen in the current object context."""
        if self.seen_keys_stack:
            return self.seen_keys_stack[-1]
        return set()

    def clone(self) -> "PathState":
        """Create a deep copy of the path state."""
        new_state = PathState()
        new_state.context_stack = [(ctx_type, list(path)) for ctx_type, path in self.context_stack]
        new_state.current_key = self.current_key
        new_state.array_index = self.array_index
        new_state.seen_keys_stack = [s.copy() for s in self.seen_keys_stack]
        return new_state


class OrigamiGenerator:
    """Generate JSON objects by autoregressive sampling from a trained model.

    This is the SINGLE implementation of generation logic. Supports various
    sampling strategies (greedy, temperature, top-k, top-p).

    Public methods:
    - `generate()`: Generate complete JSON objects from scratch (handles batching)
    - `generate_from_batch()`: Core generation loop from EncodedBatch prefix
    - `get_next_token_distribution()`: Get grammar-constrained probabilities (no sampling)

    Example:
        ```python
        generator = OrigamiGenerator(model, tokenizer)

        # Generate from scratch
        objects = generator.generate(num_samples=5, temperature=0.8)

        # Generate 10000 samples with internal batching
        objects = generator.generate(num_samples=10000, batch_size=64)
        ```
    """

    def __init__(
        self,
        model: "OrigamiModel",
        tokenizer: "JSONTokenizer",
        prevent_duplicate_keys: bool = True,
        schema: dict | None = None,
    ):
        """Initialize generator.

        Args:
            model: Trained ORIGAMI model
            tokenizer: JSONTokenizer with fitted vocabulary
            prevent_duplicate_keys: If True (default), prevent generating duplicate
                keys within the same JSON object during generation.
            schema: Optional JSON Schema dict for semantic constraints.
                When provided, schema-based masks are intersected with grammar
                masks during generation to restrict outputs by type/enum/keys.

        Note:
            The Generator uses the model's current device dynamically.
            Move the model to your desired device before calling generate().
            For standalone use, CPU is typically fastest for ORIGAMI model sizes.
        """
        from origami.training.collator import OrigamiDataCollator

        self.model = model
        self.tokenizer = tokenizer
        self.prevent_duplicate_keys = prevent_duplicate_keys
        self.model.eval()

        # Collator for batch creation (include_labels=False for inference)
        self._collator = OrigamiDataCollator(tokenizer, include_labels=False)

        # Get grammar PDA reference from model for incremental constraint application
        self._grammar_pda = model._grammar_pda

        # Schema constraints for semantic restriction.
        # Uses strict UNK settings: UNK_KEY and UNK_VALUE are blocked so the
        # model cannot escape schema constraints by generating unknown tokens.
        self._schema_pda = None
        if schema is not None:
            from origami.constraints import SchemaPDA

            self._schema_pda = SchemaPDA(
                schema,
                tokenizer.vocab,
                max_depth=model.config.max_depth,
                allow_unk_key=False,
                allow_unk_value=False,
            )

        # Check if backbone supports KV caching
        self._supports_kv_cache = self._check_kv_cache_support()

    def _check_kv_cache_support(self) -> bool:
        """Check if the model's backbone supports KV caching."""
        backbone = self.model.backbone
        # Check if backbone has past_key_values parameter
        if hasattr(backbone, "forward"):
            import inspect

            sig = inspect.signature(backbone.forward)
            return "past_key_values" in sig.parameters
        return False

    @property
    def device(self) -> torch.device:
        """Get the model's current device dynamically."""
        return next(self.model.parameters()).device

    def _get_duplicate_key_mask(
        self,
        path_states: list[PathState],
        grammar_state: tuple[Tensor, ...] | None,
    ) -> Tensor:
        """Create mask for duplicate keys that should be suppressed.

        Args:
            path_states: List of PathState for each sequence in the batch
            grammar_state: Current grammar PDA state tuple, or None

        Returns:
            Boolean tensor (batch, vocab_size) where True = should be masked out
        """
        batch_size = len(path_states)
        vocab = self.tokenizer.vocab
        device = self.device

        # Initialize mask (False = don't mask, True = mask out)
        mask = torch.zeros(batch_size, vocab.size, dtype=torch.bool, device=device)

        # Check if we're in "expecting key" state (inside object, not awaiting value)
        # We can infer this from grammar_state if available
        if grammar_state is not None:
            stack, depth, awaiting_value, _, _, _ = grammar_state
            # Get current container type
            depth_idx = (depth - 1).clamp(min=0).unsqueeze(1)
            current_container = torch.gather(stack, 1, depth_idx).squeeze(1)
            # We're expecting a key when: in object AND not awaiting value
            # STACK_OBJECT = 1 (from origami.constraints.json_grammar)
            expecting_key = (current_container == 1) & ~awaiting_value
        else:
            # Fallback: check path_states directly
            expecting_key = torch.tensor(
                [
                    bool(s.context_stack)
                    and s.context_stack[-1][0] == "object"
                    and s.current_key is None
                    for s in path_states
                ],
                device=device,
            )

        # For sequences expecting a key, mask out already-seen keys
        for i, (state, expects_key) in enumerate(
            zip(path_states, expecting_key.tolist(), strict=True)
        ):
            if expects_key:
                seen = state.get_seen_keys()
                for key_id in seen:
                    mask[i, key_id] = True

        return mask

    def _path_elements_to_schema_path(self, path_elements: list[tuple[int, int]]) -> str:
        """Convert path elements [(type, id), ...] to normalized schema path string.

        Key elements are resolved to key names via vocab, index elements
        become ``*`` wildcards.
        """
        vocab = self.tokenizer.vocab
        parts: list[str] = []
        for ptype, pid in path_elements:
            if ptype == PATH_TYPE_KEY:
                token = vocab.decode(pid)
                if isinstance(token, KeyToken):
                    parts.append(token.key)
                else:
                    return ""  # Can't resolve, fall back to root
            elif ptype == PATH_TYPE_INDEX:
                parts.append("*")
        return ".".join(parts)

    def _resolve_schema_path(self, ps: PathState) -> str | None:
        """Determine the schema path that constrains the next token.

        Based on the current path state:
        - Inside object, no key set → next is key → use object's path
        - Inside object, key set → next is value → use field path
        - Inside array → next is element → use element path (with ``*``)
        """
        if not ps.context_stack:
            return ""  # Root level

        ctx_type, base_path = ps.context_stack[-1]

        if ctx_type == "object":
            if ps.current_key is not None:
                # Next is value → use value path (includes key)
                return self._path_elements_to_schema_path(ps.get_value_path())
            else:
                # Next is key (or OBJ_END) → use object's path
                return self._path_elements_to_schema_path(base_path)
        elif ctx_type == "array":
            # Next is element (or ARRAY_END) → use element path
            return self._path_elements_to_schema_path(ps.get_value_path())

        return None

    def _compute_schema_mask(
        self,
        path_states: list[PathState],
        schema_states: list["SchemaState"],
    ) -> Tensor:
        """Compute schema constraint mask for the next token position.

        Combines path-dependent constraints (type/enum/key) from the mask table
        with count-dependent adjustments (required, minItems, maxItems).

        Args:
            path_states: Path state for each active sequence
            schema_states: Schema state for each active sequence

        Returns:
            ``(batch, vocab_size)`` boolean mask
        """
        batch_size = len(path_states)
        vocab = self.tokenizer.vocab
        device = self.device
        schema_pda = self._schema_pda

        mask_table = schema_pda.get_mask_table_on_device(device)
        mask = torch.ones(batch_size, vocab.size, dtype=torch.bool, device=device)

        for i, (ps, ss) in enumerate(zip(path_states, schema_states, strict=True)):
            # 1. Path-dependent mask from mask table
            schema_path = self._resolve_schema_path(ps)
            if schema_path is not None:
                idx = schema_pda._path_to_index.get(schema_path, 0)
                mask[i] = mask_table[idx]

            # 2. Count-dependent: required fields (suppress OBJ_END if missing)
            if ss.container_stack and ss.container_stack[-1] == "object" and ps.current_key is None:
                obj_path = (
                    self._path_elements_to_schema_path(ps.context_stack[-1][1])
                    if ps.context_stack
                    else ""
                )
                constraints = schema_pda.get_constraints(obj_path)
                if constraints and constraints.required_key_ids:
                    seen = ss.seen_keys[-1] if ss.seen_keys else set()
                    if constraints.required_key_ids - seen:
                        mask[i, vocab.obj_end_id] = False

            # 3. Count-dependent: array bounds and uniqueItems
            if ss.container_stack and ss.container_stack[-1] == "array":
                arr_path = (
                    self._path_elements_to_schema_path(ps.context_stack[-1][1])
                    if ps.context_stack
                    else ""
                )
                constraints = schema_pda.get_constraints(arr_path)
                if constraints:
                    arr_count = ss.array_counts[-1] if ss.array_counts else 0
                    if constraints.min_items is not None and arr_count < constraints.min_items:
                        # Suppress ARRAY_END until min items reached
                        mask[i, vocab.array_end_id] = False
                    if constraints.max_items is not None and arr_count >= constraints.max_items:
                        # Suppress value-like tokens to force ARRAY_END
                        for vid in vocab._value_ids:
                            mask[i, vid] = False
                        mask[i, vocab.obj_start_id] = False
                        mask[i, vocab.array_start_id] = False
                        mask[i, vocab.num_token_id] = False
                        mask[i, vocab.unk_value_id] = False

                    # uniqueItems: suppress already-seen value token IDs
                    if constraints.unique_items and ss.seen_array_values:
                        for seen_id in ss.seen_array_values[-1]:
                            mask[i, seen_id] = False

        return mask

    @torch.inference_mode()
    def generate(
        self,
        num_samples: int = 1,
        batch_size: int = 32,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        profile: bool = False,
        allow_complex_values: bool = True,
    ) -> list[dict] | tuple[list[dict], Any]:
        """Generate complete JSON objects from scratch.

        Starts from [START] token and generates until [END] is produced.
        Handles batching internally for large num_samples.

        Args:
            num_samples: Total number of objects to generate
            batch_size: Number of samples to generate in parallel
            max_length: Maximum sequence length (default: 512)
            temperature: Sampling temperature (1.0 = unchanged, <1.0 = more greedy)
            top_k: If set, only sample from top-k most likely tokens
            top_p: If set, sample from smallest set with cumulative prob >= top_p
            seed: Random seed for reproducibility
            profile: If True, profile the generation and return (results, stats)
            allow_complex_values: If False, restrict field values to primitives only
                (no nested objects or arrays). Useful for untrained models. Default True.

        Returns:
            List of generated JSON objects, or tuple of (results, pstats.Stats) if profile=True
        """
        if profile:
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()

        if seed is not None:
            torch.manual_seed(seed)

        max_length = max_length or 512

        results: list[dict] = []

        # Process in batches
        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)
            batch = self._create_start_batch(n)
            batch_results = self.generate_from_batch(
                batch,
                stop_after_value=False,
                max_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                allow_complex_values=allow_complex_values,
            )
            results.extend(batch_results)

        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            return results, stats

        return results

    @staticmethod
    def print_profile_stats(
        stats: Any,
        top_n: int = 30,
        sort_by: str = "cumulative",
    ) -> None:
        """Print profiling statistics in a readable format.

        Args:
            stats: pstats.Stats object from generate(profile=True)
            top_n: Number of top functions to show
            sort_by: Sort key - 'cumulative', 'time', 'calls', etc.
        """
        stats.strip_dirs()
        stats.sort_stats(sort_by)
        stats.print_stats(top_n)

    def _create_start_batch(self, num_samples: int) -> "EncodedBatch":
        """Create batch with START and OBJ_START tokens to generate objects.

        Args:
            num_samples: Number of sequences to create

        Returns:
            EncodedBatch with [START, OBJ_START] for each sequence
        """
        from origami.tokenizer.json_tokenizer import EncodedBatch

        vocab = self.tokenizer.vocab
        max_depth = self.tokenizer.max_depth
        seq_len = 2  # START + OBJ_START

        # Initialize with [START, OBJ_START] to ensure we generate objects
        input_ids = torch.zeros(num_samples, seq_len, dtype=torch.long, device=self.device)
        input_ids[:, 0] = vocab.start_id
        input_ids[:, 1] = vocab.obj_start_id

        # Initialize path tensors (both START and OBJ_START have empty path)
        path_types = torch.zeros(
            num_samples, seq_len, max_depth, dtype=torch.long, device=self.device
        )
        path_ids = torch.zeros(
            num_samples, seq_len, max_depth, dtype=torch.long, device=self.device
        )
        path_lengths = torch.zeros(num_samples, seq_len, dtype=torch.long, device=self.device)

        # Attention mask: all ones since no padding
        attention_mask = torch.ones(num_samples, seq_len, dtype=torch.bool, device=self.device)

        # Numeric values (empty for structural tokens)
        numeric_values = torch.zeros(num_samples, seq_len, dtype=torch.float, device=self.device)
        numeric_mask = torch.zeros(num_samples, seq_len, dtype=torch.bool, device=self.device)

        # Lengths
        lengths = torch.full((num_samples,), seq_len, dtype=torch.long, device=self.device)

        return EncodedBatch(
            input_ids=input_ids,
            path_types=path_types,
            path_ids=path_ids,
            path_lengths=path_lengths,
            attention_mask=attention_mask,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            lengths=lengths,
            labels=None,  # No labels for generation
        )

    @torch.inference_mode()
    def generate_from_batch(
        self,
        batch: "EncodedBatch",
        stop_after_value: bool = False,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        allow_complex_values: bool = True,
    ) -> list[Any]:
        """Core generation loop from EncodedBatch prefix.

        This is the SINGLE implementation of the generation loop.
        ALL generation logic lives here. Called by:
        - generate() with START-only batch, stop_after_value=False
        - Predictor with truncated batch, stop_after_value=True

        Uses dynamic batch compaction: completed sequences are removed from the
        batch to avoid unnecessary computation on finished sequences.

        Args:
            batch: Input EncodedBatch (may be left-padded prefixes)
            stop_after_value: If True, stop each sequence after completing one value
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            allow_complex_values: If False, disallow OBJ_START/ARRAY_START tokens,
                forcing primitive values only (single token). Default True.

        Returns:
            List of generated values (one per batch item).
            - If stop_after_value=True and allow_complex_values=True: primitives, lists, or dicts
            - If stop_after_value=True and allow_complex_values=False: primitives only
            - If stop_after_value=False: complete JSON objects (dicts)

        Raises:
            GenerationError: If decoding fails
        """
        vocab = self.tokenizer.vocab
        original_batch_size = batch.input_ids.size(0)

        # Ensure batch is on the right device
        batch = batch.to(self.device)

        # Clone tensors for generation (we'll extend them)
        current_ids = batch.input_ids.clone()
        current_path_types = batch.path_types.clone()
        current_path_ids = batch.path_ids.clone()
        current_path_lengths = batch.path_lengths.clone()
        current_attention_mask = batch.attention_mask.clone()

        # Track numeric values for continuous head
        current_numeric_values = batch.numeric_values.clone()

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

        # Initialize schema states for incremental constraint application
        schema_states: list[SchemaState] | None = None
        if self._schema_pda is not None:
            schema_states = []
            for i in range(original_batch_size):
                mask = current_attention_mask[i]
                seq_tokens = current_ids[i][mask].tolist()
                ss = self._schema_pda.init_state_from_tokens(seq_tokens, vocab)
                schema_states.append(ss)

        # Initialize grammar state for all sequences in parallel from their prefixes
        # init_state_from_tokens_batch returns state AFTER processing the prefix
        grammar_state = None
        initial_depths = None
        next_valid_mask = None  # Valid mask for the next token to generate
        if self._grammar_pda is not None:
            # Use batched initialization - much faster than per-sequence loop
            grammar_state = self._grammar_pda.init_state_from_tokens_batch(
                current_ids, current_attention_mask, self.device
            )
            # Record initial depths for stop_after_value
            initial_depths = grammar_state[1].clone()
            # Get valid mask for next token (state is already after processing prefix)
            next_valid_mask = self._grammar_pda._get_valid_tokens(
                grammar_state[0],  # stack
                grammar_state[1],  # depth
                grammar_state[2],  # awaiting_value
                grammar_state[3],  # seen_start
                grammar_state[4],  # root_closed
                grammar_state[5],  # ended
                self._grammar_pda._key_ids.to(self.device),
                self._grammar_pda._value_ids.to(self.device),
                self.device,
            )

        # Track where each sequence's generated content starts
        # Use sum of attention mask (count of real tokens) since we extract only
        # non-padded tokens when storing completed sequences
        gen_start_positions = batch.attention_mask.sum(dim=1).long()

        # Track original indices for each active sequence (for reordering at end)
        active_indices = list(range(original_batch_size))

        # Store completed results: maps original index -> (seq_tokens, numeric_values)
        completed_results: dict[int, tuple[list[int], list[float]]] = {}

        # KV cache for incremental generation (if backbone supports it)
        kv_cache = None
        use_kv_cache = self._supports_kv_cache

        # Generation loop
        for step in range(max_tokens):
            if len(active_indices) == 0:
                break

            batch_size = current_ids.size(0)

            # Forward pass with optional KV caching
            if use_kv_cache and step > 0 and kv_cache is not None:
                # Incremental forward: only pass the new token
                # Use the last token and its path info
                output = self.model(
                    input_ids=current_ids[:, -1:],
                    path_types=current_path_types[:, -1:, :],
                    path_ids=current_path_ids[:, -1:, :],
                    path_lengths=current_path_lengths[:, -1:],
                    attention_mask=None,  # Not needed for single token with cache
                    numeric_values=current_numeric_values[:, -1:]
                    if self.model.continuous_head is not None
                    else None,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                kv_cache = output.past_key_values
            else:
                # Full forward (first step or no caching)
                output = self.model(
                    input_ids=current_ids,
                    path_types=current_path_types,
                    path_ids=current_path_ids,
                    path_lengths=current_path_lengths,
                    attention_mask=current_attention_mask,
                    numeric_values=current_numeric_values
                    if self.model.continuous_head is not None
                    else None,
                    past_key_values=None,
                    use_cache=use_kv_cache,
                )
                if use_kv_cache:
                    kv_cache = output.past_key_values

            # Get logits for last position
            next_logits = output.logits[:, -1, :]  # (batch, vocab_size)

            # Apply grammar constraints using pre-computed valid mask
            if next_valid_mask is not None:
                next_logits = next_logits.masked_fill(~next_valid_mask, float("-inf"))

            # Apply schema constraints (type/enum/key + count-dependent)
            if self._schema_pda is not None and schema_states is not None:
                schema_mask = self._compute_schema_mask(path_states, schema_states)
                next_logits = next_logits.masked_fill(~schema_mask, float("-inf"))

            # Apply duplicate key prevention
            if self.prevent_duplicate_keys:
                dup_key_mask = self._get_duplicate_key_mask(path_states, grammar_state)
                next_logits = next_logits.masked_fill(dup_key_mask, float("-inf"))

            # Disallow complex values (objects/arrays) if requested
            if not allow_complex_values:
                next_logits[:, vocab.obj_start_id] = float("-inf")
                next_logits[:, vocab.array_start_id] = float("-inf")

            # Sample next token
            next_tokens = self._sample(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Update grammar state with sampled token and get mask for next iteration
            if self._grammar_pda is not None and grammar_state is not None:
                next_valid_mask, grammar_state = self._grammar_pda.get_next_token_mask(
                    next_tokens, grammar_state
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

                # Resolve schema min/max bounds for truncated sampling
                lower = None
                upper = None
                if self._schema_pda is not None:
                    lo_vals = torch.full((batch_size,), float("-inf"), device=self.device)
                    hi_vals = torch.full((batch_size,), float("inf"), device=self.device)
                    has_bounds = False
                    for i, ps in enumerate(path_states):
                        if not is_num[i]:
                            continue
                        schema_path = self._resolve_schema_path(ps)
                        if schema_path is not None:
                            constraints = self._schema_pda.get_constraints(schema_path)
                            if constraints:
                                if constraints.minimum is not None:
                                    lo_vals[i] = constraints.minimum
                                    has_bounds = True
                                if constraints.maximum is not None:
                                    hi_vals[i] = constraints.maximum
                                    has_bounds = True
                    if has_bounds:
                        lower = lo_vals.unsqueeze(1)  # (batch, 1) for seq_len=1
                        upper = hi_vals.unsqueeze(1)

                # Sample from MoG (truncated if bounds available)
                sampled = self.model.continuous_head.sample(
                    w.unsqueeze(1),
                    m.unsqueeze(1),
                    lv.unsqueeze(1),
                    lower=lower,
                    upper=upper,
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
                # Stop when a complete value has been generated:
                # - For objects/arrays: depth returns to initial after OBJ_END/ARRAY_END
                # - For primitives: depth stays at initial, check if we just generated a value token
                current_depths = grammar_state[1]  # depth is second element
                depth_returned = current_depths <= initial_depths

                # Check if we generated a primitive value token (not OBJ_START/ARRAY_START)
                is_container_start = (next_tokens == vocab.obj_start_id) | (
                    next_tokens == vocab.array_start_id
                )
                # Value is complete if depth returned to initial AND we didn't just start a new container
                just_completed = depth_returned & ~is_container_start
            else:
                # Stop on END token
                just_completed = next_tokens == vocab.end_id

            # Update path states and get new path tensors
            new_path_types, new_path_ids, new_path_lengths = self._update_paths(
                next_tokens, path_states, just_completed
            )

            # Update schema states with sampled tokens
            if schema_states is not None:
                for idx_s, (token_id, is_done) in enumerate(
                    zip(next_tokens.tolist(), just_completed.tolist(), strict=True)
                ):
                    if not is_done:
                        self._schema_pda.update_state(token_id, schema_states[idx_s])

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
                new_schema_states = [] if schema_states is not None else None

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
                        if new_schema_states is not None:
                            new_schema_states.append(schema_states[i])

                # Compact tensors to only keep active sequences
                if keep_indices:
                    keep_tensor = torch.tensor(keep_indices, device=self.device)
                    current_ids = current_ids[keep_tensor]
                    current_path_types = current_path_types[keep_tensor]
                    current_path_ids = current_path_ids[keep_tensor]
                    current_path_lengths = current_path_lengths[keep_tensor]
                    current_numeric_values = current_numeric_values[keep_tensor]
                    current_attention_mask = current_attention_mask[keep_tensor]

                    # Compact grammar state and valid mask
                    if grammar_state is not None:
                        grammar_state = tuple(s[keep_tensor] for s in grammar_state)
                        if initial_depths is not None:
                            initial_depths = initial_depths[keep_tensor]
                    if next_valid_mask is not None:
                        next_valid_mask = next_valid_mask[keep_tensor]

                    # Compact KV cache (select only active sequences)
                    if kv_cache is not None:
                        from origami.model.backbones import KVCache

                        new_cache = KVCache()
                        for layer_idx in range(len(kv_cache)):
                            k, v = kv_cache[layer_idx]
                            # k, v shape: (batch, num_heads, seq_len, head_dim)
                            new_cache.update(layer_idx, k[keep_tensor], v[keep_tensor])
                        kv_cache = new_cache
                else:
                    # All sequences completed, clear cache
                    kv_cache = None

                active_indices = new_active_indices
                path_states = new_path_states
                if new_schema_states is not None:
                    schema_states = new_schema_states

        # Store any remaining sequences that didn't complete (hit max_tokens)
        incomplete_indices = set()
        for i, orig_idx in enumerate(active_indices):
            mask = current_attention_mask[i]
            seq_tokens = current_ids[i][mask].tolist()
            completed_results[orig_idx] = (seq_tokens, sampled_numeric_values[orig_idx])
            incomplete_indices.add(orig_idx)

        # Decode all sequences in original order
        results = []
        for orig_idx in range(original_batch_size):
            seq, numeric_vals = completed_results[orig_idx]
            start_pos = gen_start_positions[orig_idx].item()

            if stop_after_value:
                # Check if this sequence completed properly
                if orig_idx in incomplete_indices:
                    raise GenerationError(
                        f"Sequence {orig_idx} did not complete value within max_tokens. "
                        f"The model generated {len(seq) - start_pos} value tokens without completing. "
                        f"Try increasing max_tokens or using a trained model.",
                        token_ids=seq,
                        position=len(seq),
                        vocab=vocab,
                    )
                # Decode just the generated value tokens
                value_tokens = seq[start_pos:]
                value = self._decode_value_tokens(value_tokens, numeric_vals)
                results.append(value)
            else:
                # Decode full sequence as JSON object
                # Find END token position
                try:
                    end_pos = seq.index(vocab.end_id)
                    seq = seq[: end_pos + 1]
                except ValueError:
                    # No END token - sequence didn't complete within max_tokens
                    # This happens when the model never generates proper closing tokens
                    raise GenerationError(
                        f"Sequence {orig_idx} did not complete within max_tokens. "
                        f"The model generated {len(seq)} tokens without producing END. "
                        f"Try increasing max_length or using a lower temperature.",
                        token_ids=seq,
                        vocab=vocab,
                    ) from None

                obj = self._decode_with_numerics(seq, numeric_vals)
                results.append(obj)

        return results

    @torch.inference_mode()
    def get_next_token_distribution(
        self,
        batch: "EncodedBatch",
        allow_complex_values: bool = True,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        """Get grammar-constrained probability distribution for next token.

        This is used by Predictor.predict_proba() to get the distribution over
        possible values without actually sampling/generating.

        Args:
            batch: EncodedBatch ending at position to predict
            allow_complex_values: If False, zero out OBJ_START/ARRAY_START
                probabilities and re-normalize. Default True.

        Returns:
            probs: (batch, vocab_size) probabilities after grammar masking
            continuous_params: Optional (weights, means, log_vars) for MoG head,
                               each with shape (batch, n_components)
        """
        # Ensure batch is on the right device
        batch = batch.to(self.device)

        # 1. Forward pass
        output = self.model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            numeric_values=batch.numeric_values,
        )

        # 2. Get logits at last position (predicts next token)
        next_logits = output.logits[:, -1, :]  # (batch, vocab_size)

        # 3. Apply grammar constraints
        if self._grammar_pda is not None:
            # Initialize grammar state from the full sequence (batched for efficiency)
            grammar_state = self._grammar_pda.init_state_from_tokens_batch(
                batch.input_ids, batch.attention_mask, self.device
            )

            # Get valid next tokens based on grammar state
            last_token = batch.input_ids[:, -1]
            valid_mask, _ = self._grammar_pda.get_next_token_mask(last_token, grammar_state)
            next_logits = next_logits.masked_fill(~valid_mask, float("-inf"))

        # 3b. Apply schema constraints
        if self._schema_pda is not None:
            vocab = self.tokenizer.vocab
            path_states = []
            schema_states_dist = []
            for i in range(batch.input_ids.size(0)):
                att_mask = batch.attention_mask[i]
                seq_tokens = batch.input_ids[i][att_mask].tolist()
                ps = self._init_path_states_from_tokens(seq_tokens, num_samples=1)[0]
                path_states.append(ps)
                ss = self._schema_pda.init_state_from_tokens(seq_tokens, vocab)
                schema_states_dist.append(ss)

            schema_mask = self._compute_schema_mask(path_states, schema_states_dist)
            next_logits = next_logits.masked_fill(~schema_mask, float("-inf"))

        # 4. Convert to probabilities
        probs = F.softmax(next_logits, dim=-1)

        # 5. Zero out complex value tokens if requested and re-normalize
        if not allow_complex_values:
            vocab = self.tokenizer.vocab
            probs[:, vocab.obj_start_id] = 0.0
            probs[:, vocab.array_start_id] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # 6. Get continuous params if available
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

        Raises:
            GenerationError: If decoding fails
        """
        vocab = self.tokenizer.vocab

        if not token_ids:
            raise GenerationError(
                "Empty token sequence cannot be decoded",
                token_ids=token_ids,
                vocab=vocab,
            )

        first_token = token_ids[0]

        if first_token == vocab.obj_start_id:
            # Parse object
            num_idx = [0]  # Mutable counter for tracking NUM position
            return self._parse_object_tokens(
                token_ids,
                numeric_values,
                num_idx,
                full_sequence=token_ids,
                offset=0,
            )
        elif first_token == vocab.array_start_id:
            # Parse array
            num_idx = [0]
            return self._parse_array_tokens(
                token_ids,
                numeric_values,
                num_idx,
                full_sequence=token_ids,
                offset=0,
            )
        elif first_token == vocab.num_token_id:
            # NUM token - use sampled value
            if numeric_values:
                return numeric_values[0]
            raise GenerationError(
                "NUM token found but no numeric value available",
                token_ids=token_ids,
                position=0,
                vocab=vocab,
            )
        elif first_token == vocab.unk_value_id:
            # UNK_VALUE token - return None as fallback
            return None
        else:
            # Primitive value
            token = vocab.decode(first_token)
            if isinstance(token, ValueToken):
                return token.value
            raise GenerationError(
                f"Cannot decode token {first_token} as value",
                token_ids=token_ids,
                position=0,
                vocab=vocab,
            )

    def _decode_with_numerics(
        self,
        token_ids: list[int],
        numeric_values: list[float],
    ) -> dict | list:
        """Decode a full sequence with NUM token support.

        Args:
            token_ids: Full token sequence (START...END)
            numeric_values: List of sampled numeric values for NUM tokens

        Returns:
            Decoded JSON value (object or array)

        Raises:
            GenerationError: If decoding fails
        """
        vocab = self.tokenizer.vocab

        if not token_ids:
            raise GenerationError(
                "Empty token sequence cannot be decoded",
                token_ids=token_ids,
                vocab=vocab,
            )

        # Skip START token
        pos = 0
        if token_ids[pos] == vocab.start_id:
            pos += 1

        if pos >= len(token_ids):
            raise GenerationError(
                "Empty sequence after START token",
                token_ids=token_ids,
                vocab=vocab,
            )

        # Parse based on first token
        num_idx = [0]  # Mutable counter
        if token_ids[pos] == vocab.obj_start_id:
            return self._parse_object_tokens(
                token_ids[pos:],
                numeric_values,
                num_idx,
                full_sequence=token_ids,
                offset=pos,
            )
        elif token_ids[pos] == vocab.array_start_id:
            return self._parse_array_tokens(
                token_ids[pos:],
                numeric_values,
                num_idx,
                full_sequence=token_ids,
                offset=pos,
            )
        else:
            raise GenerationError(
                f"Expected OBJ_START or ARRAY_START after START, got token {token_ids[pos]}",
                token_ids=token_ids,
                position=pos,
                vocab=vocab,
            )

    def _parse_object_tokens(
        self,
        token_ids: list[int],
        numeric_values: list[float] | None = None,
        num_idx: list[int] | None = None,
        *,
        full_sequence: list[int] | None = None,
        offset: int = 0,
    ) -> dict:
        """Parse object tokens into a dictionary."""
        vocab = self.tokenizer.vocab
        result: dict[str, Any] = {}
        pos = 1  # Skip OBJ_START

        # Track full sequence for error reporting
        if full_sequence is None:
            full_sequence = token_ids
            offset = 0

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
            value, pos = self._parse_value_at(
                token_ids,
                pos,
                numeric_values,
                num_idx,
                full_sequence=full_sequence,
                offset=offset,
            )
            result[key] = value

        return result

    def _parse_array_tokens(
        self,
        token_ids: list[int],
        numeric_values: list[float] | None = None,
        num_idx: list[int] | None = None,
        *,
        full_sequence: list[int] | None = None,
        offset: int = 0,
    ) -> list:
        """Parse array tokens into a list."""
        vocab = self.tokenizer.vocab
        result: list[Any] = []
        pos = 1  # Skip ARRAY_START

        # Track full sequence for error reporting
        if full_sequence is None:
            full_sequence = token_ids
            offset = 0

        while pos < len(token_ids):
            token_id = token_ids[pos]

            if token_id == vocab.array_end_id:
                break

            value, pos = self._parse_value_at(
                token_ids,
                pos,
                numeric_values,
                num_idx,
                full_sequence=full_sequence,
                offset=offset,
            )
            result.append(value)

        return result

    def _parse_value_at(
        self,
        token_ids: list[int],
        pos: int,
        numeric_values: list[float] | None = None,
        num_idx: list[int] | None = None,
        *,
        full_sequence: list[int] | None = None,
        offset: int = 0,
    ) -> tuple[Any, int]:
        """Parse a value starting at position pos."""
        vocab = self.tokenizer.vocab

        # Track full sequence for error reporting
        if full_sequence is None:
            full_sequence = token_ids
            offset = 0

        if pos >= len(token_ids):
            raise GenerationError(
                f"Unexpected end of tokens at position {pos}",
                token_ids=full_sequence,
                position=offset + pos,
                vocab=vocab,
            )

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
            obj = self._parse_object_tokens(
                token_ids[pos:end_pos],
                numeric_values,
                num_idx,
                full_sequence=full_sequence,
                offset=offset + pos,
            )
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
            arr = self._parse_array_tokens(
                token_ids[pos:end_pos],
                numeric_values,
                num_idx,
                full_sequence=full_sequence,
                offset=offset + pos,
            )
            return arr, end_pos

        elif token_id == vocab.num_token_id:
            # NUM token - use sampled value
            if numeric_values and num_idx is not None and num_idx[0] < len(numeric_values):
                value = numeric_values[num_idx[0]]
                num_idx[0] += 1
                return value, pos + 1
            raise GenerationError(
                "NUM token found but no numeric value available",
                token_ids=full_sequence,
                position=offset + pos,
                vocab=vocab,
            )

        elif token_id == vocab.unk_value_id:
            # UNK_VALUE token - return None as fallback value
            return None, pos + 1

        else:
            # Primitive value
            token = vocab.decode(token_id)
            if isinstance(token, ValueToken):
                return token.value, pos + 1
            raise GenerationError(
                f"Cannot decode token {token_id} as value",
                token_ids=full_sequence,
                position=offset + pos,
                vocab=vocab,
            )

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

            elif token_id == vocab.unk_key_id:
                # UNK_KEY acts like a KeyToken but is a GrammarToken
                path = state.get_current_path()
                state.set_key(PATH_TYPE_KEY, vocab.unk_key_id)

            elif (
                isinstance(token, ValueToken)
                or token_id == vocab.num_token_id
                or token_id == vocab.unk_value_id
            ):
                # Value token: path includes the key/index
                # UNK_VALUE is a GrammarToken but acts like a value
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
            elif token_id == vocab.unk_key_id:
                # UNK_KEY acts like a KeyToken but is a GrammarToken
                state.set_key(PATH_TYPE_KEY, vocab.unk_key_id)
            elif (
                isinstance(token, ValueToken)
                or token_id == vocab.num_token_id
                or token_id == vocab.unk_value_id
            ):
                # UNK_VALUE is a GrammarToken but acts like a value
                if state.context_stack and state.context_stack[-1][0] == "array":
                    state.advance_array_index()
                state.current_key = None

        # Create copies for each sample
        return [state.clone() for _ in range(num_samples)]
