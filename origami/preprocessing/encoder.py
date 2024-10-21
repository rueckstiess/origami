from collections import Counter
from enum import Enum, EnumMeta
from typing import Generator, Iterable

import numpy as np
import pandas as pd
from torch import Tensor

from origami.utils.common import ArrayStart, FieldToken, Symbol


class StreamEncoder:
    """
    This is an encoder for arbitrary (hashable) values that works in a streaming
    fashion, meaning it can incrementally add new values as they are discovered.

    It has a number of encoding / decoding methods that all use the same global
    dictionary. Use whatever is best suited for the use case.
    """

    def __init__(self, predefined: Enum | dict = {}):
        if isinstance(predefined, EnumMeta):
            predefined = {a: a.value for a in predefined}
        self._next_id = 0
        self.frozen = False
        self.tokens_to_ids = {k: v for k, v in predefined.items()}
        self.ids_to_tokens = {v: k for k, v in predefined.items()}
        self.token_freq = Counter(predefined.keys())

    @property
    def vocab_size(self):
        return len(self.tokens_to_ids)

    def __len__(self):
        return len(self.tokens_to_ids)

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get_values(self) -> list:
        """returns a list of all values that the encoder holds."""
        return list(self.tokens_to_ids.keys())

    def next_id(self) -> int:
        """returns the next unused ID."""
        while self._next_id in self.ids_to_tokens:
            self._next_id += 1
        return self._next_id

    def truncate(self, max_tokens: int) -> None:
        """truncates the encoder to the given number of tokens. Symbol and ArrayStart tokens are always included,
        and the remaining tokens are sorted by frequency."""

        if len(self.token_freq) <= max_tokens:
            # already at or below limit, do nothing
            return

        # ArrayStart and Symbol tokens always need to be included, then pick the remaining ones by frequency
        most_common_tokens = [t for t in self.tokens_to_ids.keys() if isinstance(t, (ArrayStart, Symbol, FieldToken))]
        if len(most_common_tokens) > max_tokens:
            raise ValueError(
                f"Cannot truncate encoder to {max_tokens} tokens. Symbols, ArrayStarts and FieldTokens already "
                f"require at least {len(most_common_tokens)} tokens."
            )

        remaining_space = max_tokens - len(most_common_tokens)
        # get a copy of the Counter excluding Symbol and ArrayStart tokens
        non_special_freq = Counter(
            {k: v for k, v in self.token_freq.items() if not isinstance(k, (Symbol, ArrayStart, FieldToken))}
        )
        most_common_tokens.extend([tok[0] for tok in non_special_freq.most_common(remaining_space)])

        # update dictionaries with new IDs and counter
        self.tokens_to_ids = {tok: i for i, tok in enumerate(most_common_tokens)}
        self.ids_to_tokens = {v: k for k, v in self.tokens_to_ids.items()}
        self.token_freq = Counter({k: v for k, v in self.token_freq.items() if k in most_common_tokens})

    def encode_val(self, token: any) -> int:
        """encodes a single value and returns its ID. The value must be hashable."""

        if token in self.tokens_to_ids:
            self.token_freq.update([token])
            return self.tokens_to_ids[token]

        if self.frozen:
            try:
                return self.tokens_to_ids[Symbol.UNKNOWN]
            except KeyError:
                raise KeyError(f"Encoder is frozen, key {token} not found.")

        # return new key
        id = self.next_id()
        self.tokens_to_ids[token] = id
        self.ids_to_tokens[id] = token
        self.token_freq.update([token])

        return id

    # -- Encoding --

    def encode_iter(self, tokens: Iterable) -> Generator:
        """Iterates over the tokens and returns a generator of IDs."""
        return (self.encode_val(v) for v in tokens)

    def encode_list(self, tokens: list) -> list:
        """Iterates over the list of tokens and returns a list of IDs."""
        return list(self.encode_iter(tokens))

    def encode_dict(self, d: dict, include_keys=False) -> dict:
        """Iterates over the values of the dictionary, and returns a new dictonary
        with encoded values. If `include_keys` is set to True, also encodes the
        keys of the dictionary (default is False)."""
        keys = self.encode_iter(d.keys()) if include_keys else d.keys()
        vals = self.encode_iter(d.values())

        return dict(zip(keys, vals))

    def encode(self, item: any, include_dict_keys=False) -> any:
        """recursively walks through item and encodes all values. If `include_dict_keys`
        is True, will also replace the keys in dicts."""

        # tensors, arrays, series are treated as lists (returns lists)
        if isinstance(item, (Tensor, np.ndarray, pd.Series)):
            item = item.tolist()

        # dictionaries are encoded recursively (optionally key is encoded too)
        if isinstance(item, dict):
            keys = (self.encode(k, include_dict_keys) if include_dict_keys else k for k in item.keys())
            values = (self.encode(v, include_dict_keys) for v in item.values())
            return dict(zip(keys, values))

        # lists, sets, tuples are encoded and returned as their original types
        if isinstance(item, (set, list, tuple)):
            t = type(item)
            return t(self.encode(v, include_dict_keys) for v in item)

        else:
            return self.encode_val(item)

    # -- Decoding --

    def decode_val(self, id: int) -> any:
        """decodes a single value."""
        return self.ids_to_tokens[id]

    def decode_iter(self, ids: Iterable[int]) -> Generator:
        """iterates over ids and returns a generator of decoded tokens."""
        return (self.decode_val(id) for id in ids)

    def decode_list(self, ids: list[int]) -> list:
        """iterates over the list of ids and returns a list of decoded tokens."""
        return list(self.decode_iter(ids))

    def decode_dict(self, d: dict, include_keys=False) -> dict:
        """Iterates over the values of the dictionary, and returns a new dictonary
        with decoded values. If `include_keys` is set to True, also decodes the
        keys of the dictionary (default is False)."""
        keys = self.decode_iter(d.keys()) if include_keys else d.keys()
        vals = self.decode_iter(d.values())

        return dict(zip(keys, vals))

    def decode(self, item: any, include_dict_keys=False) -> any:
        """recursively walks through item and decodes all values. If `include_dict_keys`
        is True, will also replace the keys in dicts."""

        # tensors, arrays, series are treated as lists (returns lists)
        if isinstance(item, (Tensor, np.ndarray, pd.Series)):
            item = item.tolist()

        # dictionaries are decoded recursively (optionally key is decoded too)
        if isinstance(item, dict):
            keys = (self.decode(k, include_dict_keys) if include_dict_keys else k for k in item.keys())
            values = (self.decode(v, include_dict_keys) for v in item.values())
            return dict(zip(keys, values))

        # lists, sets, tuples are decoded and returned as their original types
        if isinstance(item, (list, tuple, set)):
            t = type(item)
            return t(self.decode(v, include_dict_keys) for v in item)

        else:
            return self.decode_val(item)
