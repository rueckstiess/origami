"""Tests for LengthGroupedSampler."""

import torch

from origami.training import LengthGroupedSampler


def _batches(indices: list[int], batch_size: int) -> list[list[int]]:
    """Chunk an index stream the way a DataLoader (drop_last=False) would."""
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


class TestLengthGroupedSampler:
    def test_returns_permutation_of_all_indices(self):
        lengths = [10, 500, 30, 800, 5, 200, 60, 900]
        sampler = LengthGroupedSampler(lengths, batch_size=2, pool_mult=2)
        order = list(sampler)
        assert sorted(order) == list(range(len(lengths)))
        assert len(sampler) == len(lengths)

    def test_batches_are_length_homogeneous(self):
        # With a large pool, sorting spans the whole dataset, so each batch should
        # cover a narrow length range. Compare against random batching baseline.
        torch.manual_seed(0)
        lengths = torch.randint(100, 9000, (512,)).tolist()
        batch_size = 8

        sampler = LengthGroupedSampler(lengths, batch_size=batch_size, pool_mult=50)
        grouped = _batches(list(sampler), batch_size)

        def padding_waste(batches: list[list[int]]) -> int:
            # Tokens spent on padding = sum over batches of (max_len * n - sum(len)).
            waste = 0
            for b in batches:
                ls = [lengths[i] for i in b]
                waste += max(ls) * len(b) - sum(ls)
            return waste

        random_order = torch.randperm(len(lengths)).tolist()
        random_batches = _batches(random_order, batch_size)

        # Length grouping should cut padding waste dramatically.
        assert padding_waste(grouped) < 0.2 * padding_waste(random_batches)

    def test_longest_sequence_in_first_batch(self):
        lengths = [10, 20, 9000, 30, 40, 50, 60, 70]
        batch_size = 2
        sampler = LengthGroupedSampler(lengths, batch_size=batch_size, pool_mult=2)
        order = list(sampler)
        first_batch = order[:batch_size]
        # Index 2 holds the global maximum and must land in the first batch.
        assert 2 in first_batch

    def test_order_changes_across_epochs(self):
        lengths = list(range(200))
        gen = torch.Generator()
        gen.manual_seed(123)
        sampler = LengthGroupedSampler(lengths, batch_size=4, pool_mult=3, generator=gen)
        epoch1 = list(sampler)
        epoch2 = list(sampler)
        # Same multiset, but reshuffled order between epochs.
        assert sorted(epoch1) == sorted(epoch2)
        assert epoch1 != epoch2

    def test_reproducible_with_seeded_generator(self):
        lengths = list(range(200))

        def run():
            gen = torch.Generator()
            gen.manual_seed(42)
            return list(LengthGroupedSampler(lengths, batch_size=4, pool_mult=3, generator=gen))

        assert run() == run()

    def test_empty_dataset(self):
        sampler = LengthGroupedSampler([], batch_size=4, pool_mult=2)
        assert list(sampler) == []

    def test_pool_mult_one_still_valid(self):
        lengths = [5, 3, 8, 1, 9, 2, 7]
        sampler = LengthGroupedSampler(lengths, batch_size=2, pool_mult=1)
        assert sorted(sampler) == list(range(len(lengths)))
