"""Length-grouped sampling for efficient batching of variable-length sequences.

ORIGAMI sequence lengths can vary by an order of magnitude (e.g. 800-8600 tokens).
Because the collator pads every sequence in a batch to the longest member, a random
batch that happens to contain one long sequence forces all its (possibly short)
neighbours to be padded up to that length, and self-attention cost is O(n^2) in the
padded length. Grouping similar-length sequences into the same batch eliminates most
of that wasted padding compute.

This module implements the "sort-and-chunk pool" strategy (the same approach used by
HuggingFace's ``group_by_length``):

1. Randomly permute all indices each epoch (preserves shuffling/augmentation).
2. Split the permutation into "mega-batches" of ``pool_mult * batch_size`` samples.
3. Sort each mega-batch by length so that consecutive indices have similar length.
4. Flatten back to a single index stream; the DataLoader then chunks it into batches
   of ``batch_size``, so each batch is drawn from a narrow length range.

The ``pool_mult`` knob trades padding efficiency against per-epoch randomness:

- Larger pool  -> mega-batches span a wider length range, so after sorting each batch
  covers a *narrower* slice of lengths => tighter batches, less padding, but longer
  runs of length-ordered batches (less stochastic batch composition).
- Smaller pool -> more random batch order, but wider length spread within each batch
  => more padding. ``pool_mult=1`` degenerates to sorting within single batches only
  (almost no grouping benefit).

The single globally-longest sequence is forced into the first batch so that any
out-of-memory failure surfaces on step 1 rather than partway through an epoch.
"""

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class LengthGroupedSampler(Sampler[int]):
    """Yields indices ordered so that consecutive samples have similar length.

    Intended to be passed as the ``sampler`` argument of a ``DataLoader`` (with
    ``shuffle=False``); the DataLoader chunks the returned index stream into batches.
    A fresh random permutation is drawn on every ``__iter__`` call (i.e. every epoch),
    so key-order shuffling augmentation is preserved.

    Args:
        lengths: Per-sample sequence lengths, indexed the same as the dataset.
        batch_size: Batch size the DataLoader will chunk into. Used to size pools.
        pool_mult: Mega-batch size as a multiple of ``batch_size``. Higher values give
            tighter length grouping (less padding) at the cost of less random batch
            ordering. Default 50 (matches HuggingFace's heuristic).
        generator: Optional ``torch.Generator`` for reproducible shuffling. If provided,
            its state advances across epochs (so each epoch differs). When omitted, a
            generator is created and seeded from the global RNG, which keeps the order
            consistent across distributed ranks if the global seed is set.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        pool_mult: int = 50,
        generator: torch.Generator | None = None,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if pool_mult < 1:
            raise ValueError(f"pool_mult must be >= 1, got {pool_mult}")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.pool_mult = pool_mult

        if generator is None:
            # Derive a seed from the global RNG so that (a) all distributed ranks agree
            # when the global seed is set, and (b) the order varies between runs that
            # use different global seeds. The generator state then advances per epoch.
            generator = torch.Generator()
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            generator.manual_seed(seed)
        self.generator = generator

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self) -> Iterator[int]:
        return iter(
            _length_grouped_indices(self.lengths, self.batch_size, self.pool_mult, self.generator)
        )


def _length_grouped_indices(
    lengths: list[int],
    batch_size: int,
    pool_mult: int,
    generator: torch.Generator,
) -> list[int]:
    """Compute the length-grouped index order for one epoch."""
    n = len(lengths)
    if n == 0:
        return []

    indices = torch.randperm(n, generator=generator).tolist()

    mega_size = batch_size * pool_mult
    megabatches = [indices[i : i + mega_size] for i in range(0, n, mega_size)]
    # Sort each mega-batch by length, longest first (so the first element of each
    # mega-batch is its maximum).
    megabatches = [sorted(mb, key=lambda idx: lengths[idx], reverse=True) for mb in megabatches]

    # Force the globally-longest sequence into the very first batch so a potential OOM
    # happens on step 1 rather than mid-epoch.
    megabatch_maxima = [lengths[mb[0]] for mb in megabatches]
    max_mb = int(torch.tensor(megabatch_maxima).argmax().item())
    megabatches[0][0], megabatches[max_mb][0] = megabatches[max_mb][0], megabatches[0][0]

    return [idx for mb in megabatches for idx in mb]
