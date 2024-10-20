from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from storm_ml.model.vpda import DocumentVPDA
from storm_ml.utils import PositionEncodingMethod, count_parameters, torch_isin

from .gpt_base import GPTBase
from .positions import (
    IntegerPositionEncoding,
    NoPositionEncoding,
    SharedDocumentPositionEncoding,
)


class GPT(GPTBase):
    def __init__(self, model_config, train_config, vpda: Optional[DocumentVPDA] = None):
        """creates the following components:
        - two embeddings, for tokens and positions
        - a transformer encoder consisting of n_head layers
        - a head, mapping to the vocabulary size
        """
        super().__init__(model_config, train_config, vpda)

        # creating the embeddings layers and model
        self.token_embed = nn.Embedding(self.model_config.vocab_size, self.model_config.n_embd, padding_idx=0)

        # create position encoding layer
        match self.model_config.position_encoding:
            case PositionEncodingMethod.INTEGER:
                self.pos_encoding = IntegerPositionEncoding(
                    self.model_config.block_size,
                    self.model_config.n_embd,
                    fuse_with_mlp=self.model_config.fuse_pos_with_mlp,
                )
            case PositionEncodingMethod.KEY_VALUE:
                self.pos_encoding = SharedDocumentPositionEncoding(
                    self.token_embed,
                    fuse_with_mlp=self.model_config.fuse_pos_with_mlp,
                )
            case PositionEncodingMethod.NONE:
                self.pos_encoding = NoPositionEncoding()
            case _:
                raise ValueError(f"unknown position encoding {self.model_config.position_encoding}")

        encoder_layer = nn.TransformerEncoderLayer(
            self.model_config.n_embd,
            self.model_config.n_head,
            4 * self.model_config.n_embd,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.model_config.n_layer)
        self.head = nn.Linear(self.model_config.n_embd, self.model_config.vocab_size, bias=False)

        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # In the Naru paper, this is called Embedding Reuse. We share the weights of the
        # embedding layer with the output layer.
        # https://arxiv.org/abs/1608.05859
        if self.model_config.tie_weights:
            self.head.weight = self.token_embed.weight

        self.to(self.device)
        print("running on device", self.device)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = count_parameters(self)
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """A forward pass through the model up to the last hidden layer. Call this to get
        access to the embeddings produced by the final hidden layer."""

        b, t = idx.size()
        assert (
            t <= self.model_config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.model_config.block_size}"

        # token embeddings (b, t, n_embd)
        tok_emb = self.token_embed(idx)

        # position embeddings
        match self.model_config.position_encoding:
            case PositionEncodingMethod.INTEGER:
                x = self.pos_encoding(tok_emb)
            case PositionEncodingMethod.KEY_VALUE:
                # use the VPDA stacks for position encoding
                if self.training and self.model_config.guardrails:
                    # for guard rails we calculated the stacks on all tokens, but here we
                    # only need it for the inputs, so slicing 2 elements off the end (pop+push symbol)
                    stacks = torch.tensor(self.vpda.stacks[:, :-2]).to(self.device)
                else:
                    # if guard rails are not used, or if this is for inference calculate stacks
                    self.vpda.get_sequence_mask(idx)
                    stacks = torch.tensor(self.vpda.stacks).to(self.device)

                x = self.pos_encoding(tok_emb, stacks)
            case PositionEncodingMethod.NONE:
                x = self.pos_encoding(tok_emb)  # noop with pos_embed = NoPositionEncoding
            case _:
                raise ValueError(f"unknown position encoding {self.model_config.position_encoding}")

        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        # for now, we have to move mask to device after creating
        # due to a bug on MPS devices, see https://github.com/pytorch/pytorch/issues/116170
        mask = mask.to(self.device)

        return self.transformer(x, mask)

    def forward(self, idx, targets=None, out_mask=None):
        """forward pass through the model."""

        hidden = self.hidden(idx)
        logits = self.head(hidden)

        # set out-of-context logits to -inf, out_mask shape is (b, s, v)
        if out_mask is not None:
            logits.masked_fill_(~out_mask, -torch.inf)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if self.model_config.mask_field_token_losses:
                # mask out field tokens from loss calculation
                field_mask = torch_isin(targets, self.vpda.field_token_ids)
                targets[field_mask] = 0
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss

    def configure_optimizer(self, train_config):
        """return a standard AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        return optimizer
