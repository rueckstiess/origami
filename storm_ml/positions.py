import torch
from torch import nn


class NoPositionEncoding(nn.Module):
    """A position encoding that does nothing, just returns the token embeddings as is."""

    def forward(self, tok_emb: torch.Tensor) -> torch.Tensor:
        return tok_emb


class BasePositionEncoding(nn.Module):
    """Base class for Integer and Document position encodings. It allows for summation
    of token and position embedding or fusing the two together with a MLP."""

    def __init__(
        self,
        embedding_dim: int,
        fuse_with_mlp: bool = False,
        mlp_layer_factors: list[int] = None,
    ):
        super().__init__()
        if mlp_layer_factors is None:
            mlp_layer_factors = [2, 4, 1]

        self.fuse_with_mlp = fuse_with_mlp

        if fuse_with_mlp:
            # create an MLP to fuse the position embeddings with the token embeddings
            self.fuse_mlp = nn.Sequential()

            # start with 2x the width, concatenating pos_emb and tok_emb
            in_factor = 2
            for layer_idx, out_factor in enumerate(mlp_layer_factors):
                self.fuse_mlp.add_module(
                    f"fuse_pos_emb_layer_{layer_idx}",
                    nn.Linear(in_factor * embedding_dim, out_factor * embedding_dim),
                )
                if layer_idx < len(mlp_layer_factors) - 1:
                    self.fuse_mlp.add_module(f"fuse_pos_emb_relu{layer_idx}", nn.ReLU())
                in_factor = out_factor


class IntegerPositionEncoding(BasePositionEncoding):
    """Position encoding based on integer indexing. Each position is represented by an
    integer token which is then looked up in an embedding table."""

    def __init__(self, block_size: int, embedding_dim: int, fuse_with_mlp=True, **kwargs):
        super().__init__(embedding_dim, fuse_with_mlp=fuse_with_mlp, **kwargs)
        self.embedding = nn.Embedding(block_size, embedding_dim, padding_idx=0)

    def forward(self, tok_emb: torch.Tensor) -> torch.Tensor:
        seq_len = tok_emb.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=tok_emb.device).unsqueeze(0)
        pos_emb = self.embedding(pos)

        # now we add up the token and position embeddings, either MLP-fused or as a sum
        if self.fuse_with_mlp:
            pos_emb = pos_emb.repeat(tok_emb.size(0), 1, 1)
            return self.fuse_mlp(torch.cat((pos_emb, tok_emb), dim=2))

        return tok_emb + pos_emb


class DocumentPositionEncoding(BasePositionEncoding):
    """Position encoding based on PDA stacks. Each position is represented by a stack of
    symbols which we get from the DocumentVPDA class. The symbols are embedded and summed up
    which forms the position embedding. This can be summed with the token embeddings or
    fused via MLP (fuse_with_mlp=True)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, fuse_with_mlp=True, **kwargs):
        super().__init__(embedding_dim, fuse_with_mlp=fuse_with_mlp, **kwargs)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, tok_emb: torch.Tensor, stacks: torch.Tensor) -> torch.Tensor:
        # get mask of all negative values on the stacks
        neg_mask = stacks < 0
        zero_mask = stacks == 0

        # create embeddings of the absolute values of the stacks (since we can't embed
        # negative indices)
        embedded_stacks = self.embedding(stacks.abs())

        # now negate the embeddings for all negative positions and zero out no-ops
        embedded_stacks[neg_mask] *= -1
        embedded_stacks[zero_mask] = 0

        # calculate the cumulative sum along the sequence dimension
        embedded_stacks = embedded_stacks.cumsum(dim=1)

        # since the stacks start with one START symbol and grow by 2 columns (pop,push)
        # we skip each second column starting at 0.
        pos_emb = embedded_stacks[:, 0:-1:2, :]

        # now we add up the token and position embeddings, either MLP-fused or as a sum
        if self.fuse_with_mlp:
            return self.fuse_mlp(torch.cat((pos_emb, tok_emb), dim=2))

        return tok_emb + pos_emb


class SharedDocumentPositionEncoding(DocumentPositionEncoding):
    """This is the same as DocumentPositionEncoding but it shares the embedding layer with
    the embedding we already use for all other tokens."""

    def __init__(self, embedding: nn.Embedding, fuse_with_mlp=False, **kwargs):
        super().__init__(
            embedding.weight.size(0),
            embedding.weight.size(1),
            fuse_with_mlp=fuse_with_mlp,
            **kwargs,
        )
        self.embedding = embedding
