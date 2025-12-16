"""Tests for the KVPE (Key-Value Position Encoding) module."""

import pytest
import torch

from origami.position_encoding import (
    PATH_TYPE_INDEX,
    PATH_TYPE_KEY,
    PATH_TYPE_PAD,
    GRUPooling,
    KeyValuePositionEncoding,
    PathEncoder,
    RotaryPooling,
    SumPooling,
    TransformerPooling,
    WeightedSumPooling,
    create_pooling,
    make_depth_mask,
)
from origami.tokenizer import IndexElement, KeyElement, Vocabulary


class TestMakeDepthMask:
    """Tests for the make_depth_mask utility."""

    def test_basic_mask(self):
        """Mask should be True for indices < path_length."""
        path_lengths = torch.tensor([[2, 3, 1]])
        mask = make_depth_mask(path_lengths, max_depth=4)

        expected = torch.tensor(
            [
                [
                    [True, True, False, False],  # length 2
                    [True, True, True, False],  # length 3
                    [True, False, False, False],  # length 1
                ]
            ]
        )
        assert mask.equal(expected)

    def test_zero_length(self):
        """Zero length paths should have all False mask."""
        path_lengths = torch.tensor([[0, 2]])
        mask = make_depth_mask(path_lengths, max_depth=3)

        assert not mask[0, 0].any()  # All False for length 0
        assert mask[0, 1, :2].all()  # First 2 True for length 2

    def test_batch_shape(self):
        """Output shape should match (batch, seq_len, max_depth)."""
        path_lengths = torch.tensor([[1, 2], [3, 4]])
        mask = make_depth_mask(path_lengths, max_depth=5)
        assert mask.shape == (2, 2, 5)


class TestSumPooling:
    """Tests for sum pooling strategy."""

    def test_basic_sum(self):
        """Sum pooling should sum valid path elements."""
        pooling = SumPooling()

        # 2 positions, 3 max depth, 4 dims
        path_embeds = torch.tensor(
            [
                [
                    [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],  # position 0
                    [[4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]],  # position 1
                ]
            ],
            dtype=torch.float,
        )
        path_lengths = torch.tensor([[2, 3]])

        result = pooling(path_embeds, path_lengths)

        # Position 0: sum of first 2 elements = 1+2 = 3
        # Position 1: sum of all 3 elements = 4+5+6 = 15
        assert result.shape == (1, 2, 4)
        assert result[0, 0, 0].item() == 3
        assert result[0, 1, 0].item() == 15

    def test_sum_is_commutative(self):
        """Sum pooling should give same result regardless of element order."""
        pooling = SumPooling()

        # Same elements in different order
        embeds1 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        embeds2 = torch.tensor([[[[3.0, 4.0], [1.0, 2.0]]]])
        lengths = torch.tensor([[2]])

        result1 = pooling(embeds1, lengths)
        result2 = pooling(embeds2, lengths)

        assert torch.allclose(result1, result2)


class TestWeightedSumPooling:
    """Tests for weighted sum pooling strategy."""

    def test_learnable_weights(self):
        """Learnable weights should be trainable parameters."""
        pooling = WeightedSumPooling(max_depth=5, learnable=True)
        assert pooling.weights.requires_grad

    def test_fixed_weights(self):
        """Fixed weights should use exponential decay."""
        pooling = WeightedSumPooling(max_depth=5, learnable=False)
        assert not pooling.weights.requires_grad
        # First weight should be 1.0, second 0.9, etc.
        assert pooling.weights[0].item() == pytest.approx(1.0)
        assert pooling.weights[1].item() == pytest.approx(0.9)

    def test_forward_shape(self):
        """Output shape should match (batch, seq_len, d_model)."""
        pooling = WeightedSumPooling(max_depth=8)
        path_embeds = torch.randn(2, 5, 8, 16)
        path_lengths = torch.randint(1, 8, (2, 5))

        result = pooling(path_embeds, path_lengths)
        assert result.shape == (2, 5, 16)


class TestRotaryPooling:
    """Tests for rotary pooling strategy."""

    def test_requires_even_d_model(self):
        """Rotary pooling requires even d_model."""
        with pytest.raises(AssertionError):
            RotaryPooling(d_model=15, max_depth=8)

    def test_forward_shape(self):
        """Output shape should match (batch, seq_len, d_model)."""
        pooling = RotaryPooling(d_model=16, max_depth=8)
        path_embeds = torch.randn(2, 5, 8, 16)
        path_lengths = torch.randint(1, 8, (2, 5))

        result = pooling(path_embeds, path_lengths)
        assert result.shape == (2, 5, 16)

    def test_rotation_is_applied(self):
        """Rotary pooling should transform embeddings differently at different depths."""
        pooling = RotaryPooling(d_model=8, max_depth=4)

        # Same embedding at different depths
        embed = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        # Path with element at depth 0
        embeds1 = torch.zeros(1, 1, 4, 8)
        embeds1[0, 0, 0] = embed
        lengths1 = torch.tensor([[1]])

        # Path with element at depth 1
        embeds2 = torch.zeros(1, 1, 4, 8)
        embeds2[0, 0, 1] = embed
        lengths2 = torch.tensor([[2]])

        result1 = pooling(embeds1, lengths1)
        result2 = pooling(embeds2, lengths2)

        # Results should differ due to rotation
        assert not torch.allclose(result1, result2)


class TestGRUPooling:
    """Tests for GRU pooling strategy."""

    def test_forward_shape(self):
        """Output shape should match (batch, seq_len, d_model)."""
        pooling = GRUPooling(d_model=16, num_layers=2)
        path_embeds = torch.randn(2, 5, 8, 16)
        path_lengths = torch.randint(1, 8, (2, 5))

        result = pooling(path_embeds, path_lengths)
        assert result.shape == (2, 5, 16)

    def test_order_matters(self):
        """GRU pooling should be sensitive to element order."""
        pooling = GRUPooling(d_model=8, num_layers=1)
        pooling.eval()  # Deterministic

        # Same elements in different order
        embeds1 = torch.tensor([[[[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]]]])
        embeds2 = torch.tensor([[[[0, 1.0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0, 0]]]])
        lengths = torch.tensor([[2]])

        result1 = pooling(embeds1, lengths)
        result2 = pooling(embeds2, lengths)

        # Results should differ due to sequence order
        assert not torch.allclose(result1, result2)


class TestTransformerPooling:
    """Tests for transformer pooling strategy."""

    def test_forward_shape(self):
        """Output shape should match (batch, seq_len, d_model)."""
        pooling = TransformerPooling(d_model=16, max_depth=8, num_layers=1, num_heads=4)
        path_embeds = torch.randn(2, 5, 8, 16)
        path_lengths = torch.randint(1, 8, (2, 5))

        result = pooling(path_embeds, path_lengths)
        assert result.shape == (2, 5, 16)

    def test_has_depth_embeddings(self):
        """Transformer pooling should have learnable depth embeddings."""
        pooling = TransformerPooling(d_model=16, max_depth=8)
        assert hasattr(pooling, "depth_embedding")
        assert pooling.depth_embedding.num_embeddings == 8


class TestCreatePooling:
    """Tests for the pooling factory function."""

    def test_create_all_types(self):
        """Factory should create all pooling types."""
        for pooling_type in ["sum", "weighted", "rotary", "gru", "transformer"]:
            pooling = create_pooling(pooling_type, d_model=16, max_depth=8)
            assert pooling is not None

    def test_unknown_type_raises(self):
        """Unknown pooling type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling type"):
            create_pooling("unknown", d_model=16, max_depth=8)


class TestKeyValuePositionEncoding:
    """Tests for the main KVPE module."""

    @pytest.fixture
    def vocab(self):
        """Create a vocabulary for testing."""
        v = Vocabulary()
        v.add_key("name")
        v.add_key("age")
        v.add_key("items")
        v.add_value("Alice")
        v.freeze()
        return v

    @pytest.fixture
    def kvpe(self, vocab):
        """Create a KVPE module with separate key embeddings."""
        return KeyValuePositionEncoding(
            d_model=16,
            vocab_size=vocab.size,
            max_depth=8,
            max_array_index=64,
            pooling="sum",
            share_key_embeddings=False,
        )

    def test_forward_shape(self, kvpe):
        """Forward pass should produce correct output shape."""
        batch_size, seq_len, max_depth = 2, 5, 8
        path_types = torch.randint(0, 3, (batch_size, seq_len, max_depth))
        path_ids = torch.randint(0, 10, (batch_size, seq_len, max_depth))
        path_lengths = torch.randint(0, max_depth, (batch_size, seq_len))

        result = kvpe(path_types, path_ids, path_lengths)
        assert result.shape == (batch_size, seq_len, 16)

    def test_empty_paths(self, kvpe):
        """Zero-length paths should produce zero embeddings (after sum)."""
        path_types = torch.zeros(1, 2, 8, dtype=torch.long)
        path_ids = torch.zeros(1, 2, 8, dtype=torch.long)
        path_lengths = torch.tensor([[0, 0]])

        result = kvpe(path_types, path_ids, path_lengths)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_shared_embeddings_raises_without_setup(self):
        """Shared embeddings mode should raise if embeddings not set."""
        kvpe_shared = KeyValuePositionEncoding(
            d_model=16,
            vocab_size=100,
            max_depth=8,
            pooling="sum",
            share_key_embeddings=True,
        )

        path_types = torch.tensor([[[PATH_TYPE_KEY]]])
        path_ids = torch.tensor([[[5]]])
        path_lengths = torch.tensor([[1]])

        with pytest.raises(RuntimeError, match="Key embeddings not set"):
            kvpe_shared(path_types, path_ids, path_lengths)

    def test_shared_embeddings_works_when_set(self):
        """Shared embeddings should work after being set."""
        kvpe_shared = KeyValuePositionEncoding(
            d_model=16,
            vocab_size=100,
            max_depth=8,
            pooling="sum",
            share_key_embeddings=True,
        )

        # Set shared embeddings
        shared_embed = torch.nn.Embedding(100, 16)
        kvpe_shared.set_key_embeddings(shared_embed)

        path_types = torch.tensor([[[PATH_TYPE_KEY]]])
        path_ids = torch.tensor([[[5]]])
        path_lengths = torch.tensor([[1]])

        result = kvpe_shared(path_types, path_ids, path_lengths)
        assert result.shape == (1, 1, 16)


class TestPathEncoder:
    """Tests for the PathEncoder utility."""

    @pytest.fixture
    def vocab(self):
        """Create a vocabulary for testing."""
        v = Vocabulary()
        v.add_key("name")  # ID 10
        v.add_key("items")  # ID 11
        v.add_key("age")  # ID 12
        v.freeze()
        return v

    @pytest.fixture
    def encoder(self, vocab):
        """Create a path encoder."""
        return PathEncoder(vocab, max_depth=8, max_array_index=64)

    def test_encode_empty_path(self, encoder):
        """Empty path should have zero length."""
        paths = [()]
        path_types, path_ids, path_lengths = encoder.encode_paths(paths)

        assert path_lengths[0].item() == 0
        assert (path_types[0] == PATH_TYPE_PAD).all()

    def test_encode_key_path(self, encoder):
        """Key elements should be encoded correctly."""
        paths = [(KeyElement("name"),)]
        path_types, path_ids, path_lengths = encoder.encode_paths(paths)

        assert path_lengths[0].item() == 1
        assert path_types[0, 0].item() == PATH_TYPE_KEY
        assert path_ids[0, 0].item() == 10  # "name" is ID 10

    def test_encode_index_path(self, encoder):
        """Index elements should be encoded correctly."""
        paths = [(KeyElement("items"), IndexElement(5))]
        path_types, path_ids, path_lengths = encoder.encode_paths(paths)

        assert path_lengths[0].item() == 2
        assert path_types[0, 0].item() == PATH_TYPE_KEY
        assert path_types[0, 1].item() == PATH_TYPE_INDEX
        assert path_ids[0, 1].item() == 5

    def test_encode_complex_path(self, encoder):
        """Complex nested paths should encode correctly."""
        # items[2].name
        paths = [(KeyElement("items"), IndexElement(2), KeyElement("name"))]
        path_types, path_ids, path_lengths = encoder.encode_paths(paths)

        assert path_lengths[0].item() == 3
        assert path_types[0, 0].item() == PATH_TYPE_KEY
        assert path_types[0, 1].item() == PATH_TYPE_INDEX
        assert path_types[0, 2].item() == PATH_TYPE_KEY

    def test_encode_batch(self, encoder):
        """Batch encoding should pad shorter sequences."""
        batch_paths = [
            [(), (KeyElement("name"),)],  # seq_len=2
            [(), (KeyElement("items"),), (KeyElement("items"), IndexElement(0))],  # seq_len=3
        ]
        path_types, path_ids, path_lengths = encoder.encode_batch(batch_paths)

        assert path_types.shape == (2, 3, 8)  # (batch, max_seq, max_depth)
        assert path_lengths.shape == (2, 3)

        # First sequence has 2 positions, second has 3
        assert path_lengths[0, 0].item() == 0  # empty path
        assert path_lengths[0, 1].item() == 1  # (name,)
        assert path_lengths[1, 2].item() == 2  # (items, [0])

    def test_truncates_long_paths(self, encoder):
        """Paths longer than max_depth should be truncated."""
        long_path = tuple(KeyElement(f"k{i}") for i in range(20))
        paths = [long_path]

        path_types, path_ids, path_lengths = encoder.encode_paths(paths)

        assert path_lengths[0].item() == 8  # Truncated to max_depth

    def test_clamps_large_indices(self, encoder):
        """Array indices >= max_array_index should be clamped."""
        paths = [(IndexElement(1000),)]  # Larger than max_array_index=64
        path_types, path_ids, path_lengths = encoder.encode_paths(paths)

        assert path_ids[0, 0].item() == 63  # Clamped to max - 1


class TestKVPEIntegration:
    """Integration tests for KVPE with tokenizer paths."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary from sample object."""
        from origami.tokenizer import JSONTokenizer

        tokenizer = JSONTokenizer()
        tokenizer.fit([{"user": {"name": "Alice"}, "items": [1, 2, 3]}])
        tokenizer.vocab.freeze()
        return tokenizer.vocab

    def test_tokenizer_paths_to_kvpe(self, vocab):
        """Paths from tokenizer should work with KVPE."""
        from origami.tokenizer import JSONTokenizer

        tokenizer = JSONTokenizer(vocab=vocab)
        instance = tokenizer.tokenize({"user": {"name": "Alice"}})

        # Create encoder and KVPE
        encoder = PathEncoder(vocab, max_depth=8)
        kvpe = KeyValuePositionEncoding(
            d_model=16,
            vocab_size=vocab.size,
            max_depth=8,
            pooling="sum",
            share_key_embeddings=False,
        )

        # Encode paths
        path_types, path_ids, path_lengths = encoder.encode_paths(instance.paths)
        path_types = path_types.unsqueeze(0)  # Add batch dim
        path_ids = path_ids.unsqueeze(0)
        path_lengths = path_lengths.unsqueeze(0)

        # Get position embeddings
        pos_embeds = kvpe(path_types, path_ids, path_lengths)

        assert pos_embeds.shape == (1, len(instance.tokens), 16)

    def test_different_paths_different_embeddings(self, vocab):
        """Different paths should produce different position embeddings."""
        encoder = PathEncoder(vocab, max_depth=8)
        kvpe = KeyValuePositionEncoding(
            d_model=16,
            vocab_size=vocab.size,
            max_depth=8,
            pooling="rotary",  # Order-aware
            share_key_embeddings=False,
        )

        # Two different paths
        paths1 = [(KeyElement("user"), KeyElement("name"))]
        paths2 = [(KeyElement("items"), IndexElement(0))]

        pt1, pi1, pl1 = encoder.encode_paths(paths1)
        pt2, pi2, pl2 = encoder.encode_paths(paths2)

        embed1 = kvpe(pt1.unsqueeze(0), pi1.unsqueeze(0), pl1.unsqueeze(0))
        embed2 = kvpe(pt2.unsqueeze(0), pi2.unsqueeze(0), pl2.unsqueeze(0))

        # Embeddings should differ
        assert not torch.allclose(embed1, embed2)

    def test_sum_pooling_is_commutative(self, vocab):
        """Sum pooling should give same result for reversed paths."""
        encoder = PathEncoder(vocab, max_depth=8)
        kvpe = KeyValuePositionEncoding(
            d_model=16,
            vocab_size=vocab.size,
            max_depth=8,
            pooling="sum",
            share_key_embeddings=False,
        )

        # Note: With sum pooling, a.b = b.a (which is a limitation)
        # This test verifies that limitation exists
        paths1 = [(KeyElement("user"), KeyElement("name"))]
        paths2 = [(KeyElement("name"), KeyElement("user"))]  # Reversed

        pt1, pi1, pl1 = encoder.encode_paths(paths1)
        pt2, pi2, pl2 = encoder.encode_paths(paths2)

        embed1 = kvpe(pt1.unsqueeze(0), pi1.unsqueeze(0), pl1.unsqueeze(0))
        embed2 = kvpe(pt2.unsqueeze(0), pi2.unsqueeze(0), pl2.unsqueeze(0))

        # Sum pooling is commutative - these should be equal!
        assert torch.allclose(embed1, embed2, atol=1e-5)

    def test_rotary_pooling_is_non_commutative(self, vocab):
        """Rotary pooling should distinguish path order."""
        encoder = PathEncoder(vocab, max_depth=8)
        kvpe = KeyValuePositionEncoding(
            d_model=16,
            vocab_size=vocab.size,
            max_depth=8,
            pooling="rotary",
            share_key_embeddings=False,
        )

        paths1 = [(KeyElement("user"), KeyElement("name"))]
        paths2 = [(KeyElement("name"), KeyElement("user"))]

        pt1, pi1, pl1 = encoder.encode_paths(paths1)
        pt2, pi2, pl2 = encoder.encode_paths(paths2)

        embed1 = kvpe(pt1.unsqueeze(0), pi1.unsqueeze(0), pl1.unsqueeze(0))
        embed2 = kvpe(pt2.unsqueeze(0), pi2.unsqueeze(0), pl2.unsqueeze(0))

        # Rotary pooling should NOT be commutative
        assert not torch.allclose(embed1, embed2, atol=1e-5)

    def test_gru_pooling_is_non_commutative(self, vocab):
        """GRU pooling should distinguish path order."""
        encoder = PathEncoder(vocab, max_depth=8)
        kvpe = KeyValuePositionEncoding(
            d_model=16,
            vocab_size=vocab.size,
            max_depth=8,
            pooling="gru",
            share_key_embeddings=False,
        )
        kvpe.eval()  # Deterministic

        paths1 = [(KeyElement("user"), KeyElement("name"))]
        paths2 = [(KeyElement("name"), KeyElement("user"))]

        pt1, pi1, pl1 = encoder.encode_paths(paths1)
        pt2, pi2, pl2 = encoder.encode_paths(paths2)

        embed1 = kvpe(pt1.unsqueeze(0), pi1.unsqueeze(0), pl1.unsqueeze(0))
        embed2 = kvpe(pt2.unsqueeze(0), pi2.unsqueeze(0), pl2.unsqueeze(0))

        # GRU pooling should NOT be commutative
        assert not torch.allclose(embed1, embed2, atol=1e-5)
