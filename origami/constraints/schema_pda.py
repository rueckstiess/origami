"""Schema-aware constraint engine with pre-computed mask table.

Provides semantic constraints on top of the syntactic JSON grammar PDA.
The schema restricts which tokens are valid at each position based on
field type, enum values, and key restrictions (additionalProperties).

Performance design:
- Path-dependent constraints (type, enum, additionalProperties) are
  pre-computed into a mask table during __init__. During collation,
  a single tensor gather produces the full schema mask — O(1) per position.
- Count-dependent constraints (minItems, maxItems, required) are stored
  as metadata and enforced only during inference in the generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from origami.tokenizer.path import IndexElement, KeyElement, Path
from origami.tokenizer.vocabulary import KeyToken, ValueToken, Vocabulary


@dataclass
class CompiledFieldConstraints:
    """Compiled constraints for a single schema path.

    Stores inference-only metadata that cannot be pre-computed
    into the mask table (count-dependent constraints).
    """

    min_items: int | None = None
    max_items: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    required_key_ids: frozenset[int] | None = None
    unique_items: bool = False


@dataclass
class SchemaState:
    """Per-sequence state for incremental inference.

    Tracks count-dependent information needed for minItems/maxItems,
    required field enforcement, and uniqueItems during generation.
    """

    # Stack of array element counts, one per array depth
    array_counts: list[int] = field(default_factory=list)
    # Stack of seen key token IDs, one per object depth
    seen_keys: list[set[int]] = field(default_factory=list)
    # Stack of container types ("object" or "array")
    container_stack: list[str] = field(default_factory=list)
    # Stack of seen value token IDs in arrays (for uniqueItems), one per array depth
    seen_array_values: list[set[int]] = field(default_factory=list)

    def push_object(self) -> None:
        self.container_stack.append("object")
        self.seen_keys.append(set())

    def push_array(self) -> None:
        self.container_stack.append("array")
        self.array_counts.append(0)
        self.seen_array_values.append(set())

    def pop_context(self) -> str | None:
        if not self.container_stack:
            return None
        ctx = self.container_stack.pop()
        if ctx == "object":
            if self.seen_keys:
                self.seen_keys.pop()
        elif ctx == "array":
            if self.array_counts:
                self.array_counts.pop()
            if self.seen_array_values:
                self.seen_array_values.pop()
        return ctx

    def record_key(self, key_id: int) -> None:
        if self.seen_keys:
            self.seen_keys[-1].add(key_id)

    def record_array_value(self, token_id: int) -> None:
        """Record a value token seen in the current array."""
        if self.seen_array_values:
            self.seen_array_values[-1].add(token_id)

    def increment_array_count(self) -> None:
        if self.array_counts:
            self.array_counts[-1] += 1

    def clone(self) -> SchemaState:
        return SchemaState(
            array_counts=list(self.array_counts),
            seen_keys=[s.copy() for s in self.seen_keys],
            container_stack=list(self.container_stack),
            seen_array_values=[s.copy() for s in self.seen_array_values],
        )


class SchemaPDA:
    """Schema-aware constraint engine with pre-computed mask table.

    Compiles a JSON Schema into a mask table where each row is a
    ``(vocab_size,)`` boolean mask for a specific schema path.
    Row 0 is all-True (no constraint). Subsequent rows correspond to
    unique schema paths with their type/enum/key restrictions.

    During training (collation), schema masks are produced by a single
    tensor gather: ``mask_table[indices]``.

    During inference, the same mask table is used for path-dependent
    constraints, with additional count-dependent enforcement applied
    on top.

    Args:
        schema: JSON Schema dict (draft 2020-12 subset)
        vocab: Fitted Vocabulary instance
        max_depth: Maximum nesting depth
        allow_unk_key: If True, UNK_KEY is allowed even when
            additionalProperties is false. Use True for evaluation
            (prevents inf loss on unseen keys), False for generation
            (prevents schema escape). Default False.
        allow_unk_value: If True, UNK_VALUE is allowed at enum/type-
            constrained positions. Use True for evaluation (prevents
            inf loss on unseen values), False for generation (forces
            the model to commit to known values). Default True.
    """

    def __init__(
        self,
        schema: dict,
        vocab: Vocabulary,
        max_depth: int = 32,
        allow_unk_key: bool = False,
        allow_unk_value: bool = True,
    ):
        self._vocab = vocab
        self._vocab_size = vocab.size
        self._max_depth = max_depth
        self._schema = schema
        self._allow_unk_key = allow_unk_key
        self._allow_unk_value = allow_unk_value

        # Build type → token ID mapping
        self._type_to_token_ids = self._build_type_to_token_ids()

        # Compile schema into mask table
        self._path_to_index: dict[str, int] = {}
        self._path_constraints: dict[str, CompiledFieldConstraints] = {}
        self._path_masks: list[Tensor] = []  # masks for rows 1..N

        self._compile_schema(schema, "")

        # Build mask table: row 0 = all-True (default), rows 1..N = path masks
        all_true = torch.ones(self._vocab_size, dtype=torch.bool)
        rows = [all_true] + self._path_masks
        self._mask_table = torch.stack(rows)  # (num_paths + 1, vocab_size)

        # Cache for device-specific mask tables
        self._device_tables: dict[torch.device, Tensor] = {}

    def _build_type_to_token_ids(self) -> dict[str, set[int]]:
        """Map JSON Schema type names to sets of vocab token IDs."""
        vocab = self._vocab
        mapping: dict[str, set[int]] = {
            "string": set(),
            "integer": set(),
            "number": set(),
            "boolean": set(),
            "null": set(),
            "object": {vocab.obj_start_id},
            "array": {vocab.array_start_id},
        }

        for token_id in vocab._value_ids:
            token = vocab.decode(token_id)
            assert isinstance(token, ValueToken)
            value = token.value

            if isinstance(value, bool):
                mapping["boolean"].add(token_id)
            elif isinstance(value, int):
                mapping["integer"].add(token_id)
                mapping["number"].add(token_id)
            elif isinstance(value, float):
                mapping["number"].add(token_id)
            elif isinstance(value, str):
                mapping["string"].add(token_id)
            elif value is None:
                mapping["null"].add(token_id)

        # NUM token is valid for numeric types
        mapping["number"].add(vocab.num_token_id)
        mapping["integer"].add(vocab.num_token_id)

        return mapping

    def _compile_schema(self, schema_node: dict, path: str) -> None:
        """Recursively compile schema into mask table rows.

        For each unique path, computes a combined mask that handles:
        - Value constraints (type/enum): restricts value-like tokens
        - Key constraints (additionalProperties): restricts key tokens
        """
        # Compute combined mask for this path
        value_mask = self._compute_value_mask(schema_node)
        key_mask = self._compute_key_mask(schema_node)
        combined = value_mask & key_mask

        # Store mask and index
        idx = len(self._path_masks) + 1  # 0 is reserved for all-True
        self._path_to_index[path] = idx
        self._path_masks.append(combined)

        # Store inference-only constraints
        constraints = CompiledFieldConstraints(
            min_items=schema_node.get("minItems"),
            max_items=schema_node.get("maxItems"),
            minimum=schema_node.get("minimum"),
            maximum=schema_node.get("maximum"),
            unique_items=schema_node.get("uniqueItems", False),
        )

        # Pre-compute required key IDs for inference
        if "required" in schema_node and "properties" in schema_node:
            required_ids = set()
            for key in schema_node["required"]:
                kid = self._key_name_to_id(key)
                if kid is not None:
                    required_ids.add(kid)
            if required_ids:
                constraints.required_key_ids = frozenset(required_ids)

        self._path_constraints[path] = constraints

        # Recurse into object properties
        if "properties" in schema_node:
            for key, sub_schema in schema_node["properties"].items():
                sub_path = f"{path}.{key}" if path else key
                self._compile_schema(sub_schema, sub_path)

        # Recurse into array items
        if "items" in schema_node:
            sub_path = f"{path}.*" if path else "*"
            self._compile_schema(schema_node["items"], sub_path)

    def _compute_value_mask(self, schema_node: dict) -> Tensor:
        """Compute mask restricting value-like tokens based on type/enum.

        Tokens that are NOT value-like (grammar tokens, key tokens) are
        left as True — the grammar mask handles those.
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool)
        vocab = self._vocab

        schema_enum = schema_node.get("enum")
        schema_type = schema_node.get("type")

        if schema_enum is None and schema_type is None:
            return mask  # No value constraint

        # Identify all "value-like" token IDs
        # These are the tokens that could appear at a value position
        value_like_ids = set(vocab._value_ids)
        value_like_ids.add(vocab.unk_value_id)
        value_like_ids.add(vocab.num_token_id)
        value_like_ids.add(vocab.obj_start_id)
        value_like_ids.add(vocab.array_start_id)

        # Disable all value-like tokens first
        for vid in value_like_ids:
            mask[vid] = False

        if schema_enum is not None:
            # Enum constraint: allow only specific values
            for value in schema_enum:
                token = ValueToken(value)
                if token in vocab._token_to_id:
                    mask[vocab._token_to_id[token]] = True
            # Allow UNK_VALUE as fallback for unseen values (e.g., in eval data).
            # Disabled during generation to prevent escape from schema constraints.
            if self._allow_unk_value:
                mask[vocab.unk_value_id] = True

            # If enum includes only primitives, don't allow containers
            # But if we also have type constraints, check for object/array
            if schema_type is not None:
                types = schema_type if isinstance(schema_type, list) else [schema_type]
                if "object" in types:
                    mask[vocab.obj_start_id] = True
                if "array" in types:
                    mask[vocab.array_start_id] = True

        elif schema_type is not None:
            # Type constraint: allow tokens matching the specified type(s)
            types = schema_type if isinstance(schema_type, list) else [schema_type]

            for t in types:
                if t in self._type_to_token_ids:
                    for tid in self._type_to_token_ids[t]:
                        mask[tid] = True

            if self._allow_unk_value:
                mask[vocab.unk_value_id] = True

        return mask

    def _compute_key_mask(self, schema_node: dict) -> Tensor:
        """Compute mask restricting key tokens based on additionalProperties.

        Only restricts key tokens. All other tokens are left as True.
        """
        mask = torch.ones(self._vocab_size, dtype=torch.bool)
        vocab = self._vocab

        additional_props = schema_node.get("additionalProperties", True)
        if additional_props is not False or "properties" not in schema_node:
            return mask  # No key constraint

        # Restrict keys to only those defined in properties.
        # By default, UNK_KEY is blocked — additionalProperties: false means
        # only known keys are valid. When allow_unk_key=True (e.g., during
        # evaluation), UNK_KEY is permitted so eval data with unseen keys
        # doesn't produce inf loss. During generation, allow_unk_key=False
        # prevents the model from escaping schema constraints via UNK_KEY.
        all_key_ids = vocab.get_all_key_ids()  # includes UNK_KEY
        allowed_key_ids: set[int] = set()

        for key in schema_node["properties"]:
            kid = self._key_name_to_id(key)
            if kid is not None:
                allowed_key_ids.add(kid)

        if self._allow_unk_key:
            allowed_key_ids.add(vocab.unk_key_id)

        # Disable disallowed keys
        for kid in all_key_ids:
            if kid not in allowed_key_ids:
                mask[kid] = False

        return mask

    def _key_name_to_id(self, key: str) -> int | None:
        """Look up the vocab ID for a key name, or None if not found."""
        token = KeyToken(key)
        return self._vocab._token_to_id.get(token)

    # --- Path normalization ---

    def normalize_path(self, path: Path) -> str:
        """Convert tokenizer Path to normalized schema path string.

        Array indices are replaced with ``*`` wildcards so all elements
        at the same array position share constraints.

        Examples:
            ``()`` → ``""``
            ``(KeyElement("name"),)`` → ``"name"``
            ``(KeyElement("items"), IndexElement(0))`` → ``"items.*"``
            ``(KeyElement("items"), IndexElement(2), KeyElement("price"))`` → ``"items.*.price"``
        """
        parts: list[str] = []
        for element in path:
            if isinstance(element, KeyElement):
                parts.append(element.key)
            elif isinstance(element, IndexElement):
                parts.append("*")
        return ".".join(parts)

    # --- Training path (mask table gather) ---

    def resolve_mask_indices(
        self,
        batch_paths: list[list[Path]],
        batch_size: int,
        seq_len: int,
        start_positions: list[int],
    ) -> Tensor:
        """Convert batch of token paths to mask table row indices.

        For each position t, looks up the schema constraint for the
        NEXT token (t+1) based on its path. This aligns with autoregressive
        semantics: logits[t] predict token at t+1.

        Args:
            batch_paths: List of path sequences (one per batch item)
            batch_size: Batch size (including padding)
            seq_len: Padded sequence length
            start_positions: Left-padding offset per batch item

        Returns:
            ``(batch_size, seq_len)`` tensor of mask table row indices.
            Index 0 = no constraint (all-True default).
        """
        indices = torch.zeros(batch_size, seq_len, dtype=torch.long)

        for b, (paths, start_pos) in enumerate(zip(batch_paths, start_positions, strict=True)):
            for t in range(len(paths)):
                pos = start_pos + t  # Position in padded sequence
                # Look up constraint for the NEXT token's path
                if t + 1 < len(paths):
                    next_path = paths[t + 1]
                    norm = self.normalize_path(next_path)
                    indices[b, pos] = self._path_to_index.get(norm, 0)
                # else: last position → index 0 (all-True)

        return indices

    def gather_masks(self, indices: Tensor) -> Tensor:
        """Look up mask table rows by index.

        Single tensor operation: ``mask_table[indices]``.

        Args:
            indices: ``(batch, seq_len)`` int tensor of row indices

        Returns:
            ``(batch, seq_len, vocab_size)`` boolean mask
        """
        return self._mask_table[indices]

    def get_mask_table_on_device(self, device: torch.device) -> Tensor:
        """Get mask table on the specified device (cached)."""
        if device not in self._device_tables:
            self._device_tables[device] = self._mask_table.to(device)
        return self._device_tables[device]

    # --- Inference path ---

    def get_mask_for_schema_path(self, schema_path: str) -> Tensor:
        """Get the pre-computed mask for a schema path.

        Args:
            schema_path: Normalized schema path string

        Returns:
            ``(vocab_size,)`` boolean mask. All-True if path not in schema.
        """
        idx = self._path_to_index.get(schema_path, 0)
        return self._mask_table[idx]

    def get_constraints(self, schema_path: str) -> CompiledFieldConstraints | None:
        """Get compiled constraints for a schema path.

        Returns None if the path has no schema constraints.
        """
        return self._path_constraints.get(schema_path)

    def init_state(self, batch_size: int) -> list[SchemaState]:
        """Create fresh schema states for a batch."""
        return [SchemaState() for _ in range(batch_size)]

    def init_state_from_tokens(
        self,
        token_ids: list[int],
        vocab: Vocabulary,
    ) -> SchemaState:
        """Initialize schema state by replaying a token sequence.

        Updates container stack and counts based on the prefix tokens.
        """
        state = SchemaState()

        for token_id in token_ids:
            if token_id == vocab.pad_token_id:
                continue
            self.update_state(token_id, state)

        return state

    def update_state(
        self,
        token_id: int,
        state: SchemaState,
    ) -> None:
        """Update schema state with a single generated token (in-place).

        Array element counting: each top-level element in an array counts as one
        item toward minItems/maxItems. For primitive arrays, each value token is
        one element. For arrays of objects/arrays, OBJ_START/ARRAY_START marks
        a new element. Values inside nested containers do NOT count toward the
        parent array.
        """
        vocab = self._vocab
        if token_id == vocab.obj_start_id:
            # Object starting directly inside an array = one array element
            if state.container_stack and state.container_stack[-1] == "array":
                state.increment_array_count()
            state.push_object()
        elif token_id == vocab.obj_end_id:
            state.pop_context()
        elif token_id == vocab.array_start_id:
            # Sub-array starting directly inside an array = one array element
            if state.container_stack and state.container_stack[-1] == "array":
                state.increment_array_count()
            state.push_array()
        elif token_id == vocab.array_end_id:
            state.pop_context()
        elif vocab.is_key_token(token_id):
            state.record_key(token_id)
        elif vocab.is_value_token(token_id):
            # Only count primitive values as array elements when directly in array
            if state.container_stack and state.container_stack[-1] == "array":
                state.record_array_value(token_id)
                state.increment_array_count()

    # --- Properties ---

    @property
    def schema(self) -> dict:
        """The original schema dict."""
        return self._schema

    @property
    def num_paths(self) -> int:
        """Number of compiled schema paths (excluding the all-True default)."""
        return len(self._path_masks)

    @property
    def mask_table(self) -> Tensor:
        """The pre-computed mask table ``(num_paths + 1, vocab_size)``."""
        return self._mask_table

    def get_compiled_paths(self) -> dict[str, int]:
        """Return mapping of schema paths to mask table row indices."""
        return dict(self._path_to_index)

    def summary(self) -> str:
        """Human-readable summary of compiled schema constraints."""
        lines = [
            f"SchemaPDA ({self.num_paths} paths, vocab_size={self._vocab_size})",
        ]
        for path in sorted(self._path_to_index.keys()):
            idx = self._path_to_index[path]
            mask = self._mask_table[idx]
            n_allowed = mask.sum().item()
            constraint = self._path_constraints.get(path)
            extra = ""
            if constraint:
                parts = []
                if constraint.minimum is not None:
                    parts.append(f"min={constraint.minimum}")
                if constraint.maximum is not None:
                    parts.append(f"max={constraint.maximum}")
                if constraint.min_items is not None:
                    parts.append(f"minItems={constraint.min_items}")
                if constraint.max_items is not None:
                    parts.append(f"maxItems={constraint.max_items}")
                if constraint.unique_items:
                    parts.append("uniqueItems")
                if constraint.required_key_ids:
                    parts.append(f"required={len(constraint.required_key_ids)} keys")
                if parts:
                    extra = f" [{', '.join(parts)}]"

            path_display = path or "<root>"
            lines.append(f"  {path_display}: {n_allowed}/{self._vocab_size} tokens{extra}")

        return "\n".join(lines)
