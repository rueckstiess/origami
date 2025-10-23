from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from mdbrtools.schema import Schema
from torch.nn.utils.rnn import pad_sequence

from origami.model import ORIGAMI
from origami.model.vpda import ObjectVPDA
from origami.preprocessing import StreamEncoder, detokenize
from origami.utils import Symbol


class Sampler:
    """Generate unbiased samples from the learned model distribution.

    Samples complete documents by starting from START token and generating
    autoregressively using the model's learned probability distribution.

    Args:
        model: Trained ORiGAMi model
        encoder: Token encoder
        schema: Optional schema for VPDA guardrails
        temperature: Sampling temperature (1.0 = model distribution, lower = more peaked)

    Example:
        >>> sampler = Sampler(model, encoder, schema)
        >>> documents, log_probs = sampler.sample(n=100)
        >>> # documents: list of dicts
        >>> # log_probs: numpy array of log P(document)
    """

    def __init__(
        self,
        model: ORIGAMI,
        encoder: StreamEncoder,
        schema: Optional[Schema] = None,
        temperature: float = 1.0,
    ):
        self.model = model
        self.encoder = encoder
        self.schema = schema
        self.temperature = temperature

        # Initialize VPDA if schema provided
        if schema is not None:
            self.vpda = ObjectVPDA(encoder, schema)
        else:
            self.vpda = None

    @torch.no_grad()
    def sample(self, n: int) -> tuple[list[dict], np.ndarray]:
        """Sample n documents from the model distribution.

        Args:
            n: Number of documents to sample

        Returns:
            Tuple of (documents, log_probabilities):
                - documents: List of n sampled documents (dicts)
                - log_probabilities: Numpy array of log P(document) for each sample

        Example:
            >>> docs, log_probs = sampler.sample(100)
            >>> probs = np.exp(log_probs)  # Convert to probabilities
            >>> cardinality_estimate = n * np.mean(probs)
        """
        self.model.eval()

        # Create batch of START tokens
        start_token = self.encoder.encode(Symbol.START)
        idx = torch.full((n, 1), start_token, dtype=torch.long)

        # Sample completions with probability tracking
        completed_idx, log_probs = self._sample_batch(idx)

        # Decode token sequences to documents
        token_sequences = self.encoder.decode(completed_idx)
        documents = [detokenize(tokens) for tokens in token_sequences]

        # Switch model back to train mode
        self.model.train()

        return documents, log_probs

    def _sample_batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        """Sample completions for a batch of sequences.

        Uses stochastic sampling from the model's distribution and tracks
        log probabilities for each generated token.

        Args:
            idx: Initial token sequences (batch_size, seq_len)

        Returns:
            Tuple of (completed_sequences, log_probabilities):
                - completed_sequences: Completed token sequences (batch_size, block_size)
                - log_probabilities: Log P(sequence) for each sample (batch_size,)
        """
        pad_token_id = self.encoder.encode(Symbol.PAD)
        guardrails = self.model.model_config.guardrails
        block_size = self.model.model_config.block_size
        device = self.model.device

        batch_size = idx.size(0)
        idx = idx.to(device)

        # Track original indices for reordering after early stopping
        indexes = torch.arange(batch_size, device=device)

        # Initialize log probability accumulator for each sequence
        log_probs = torch.zeros(batch_size, device=device)

        # Initialize VPDA if using guardrails
        if guardrails and self.vpda is not None:
            self.vpda.initialize(batch_size)
            masks = torch.tensor(self.vpda.get_sequence_mask(idx)[:, 1:])
            next_mask = torch.tensor(self.vpda.get_input_mask()).unsqueeze(1)
            masks = torch.cat((masks, next_mask), dim=1)
            masks = masks.to(device)
        else:
            masks = None

        # Track completed sequences: (original_index, sequence, log_prob)
        completed_rows = []

        # Autoregressive generation loop
        for _ in range(block_size - idx.size(1)):
            # Forward pass to get logits
            logits, _ = self.model(idx, None, masks)

            # Get logits for next token (last position in sequence)
            logits = logits[:, -1, :] / self.temperature

            # Apply guardrail mask if enabled
            if guardrails and masks is not None:
                # Mask invalid tokens by setting logits to -inf
                mask = masks[:, -1, :]
                logits = logits.masked_fill(~mask, float("-inf"))

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token from distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Track log probability of sampled token
            token_log_probs = torch.log(probs.gather(1, idx_next)).squeeze(1)
            log_probs += token_log_probs

            # Append sampled token to sequences
            idx = torch.cat((idx, idx_next), dim=1)

            # Check for completed sequences (reached PAD token)
            complete = (idx_next == pad_token_id).view(-1)

            if complete.any():
                # Save completed sequences with their indices and log probs
                for i, is_complete in enumerate(complete):
                    if is_complete:
                        completed_rows.append((indexes[i].item(), idx[i], log_probs[i].item()))

                # Remove completed sequences from batch BEFORE checking if all done
                idx = idx[~complete, :]
                indexes = indexes[~complete]
                idx_next = idx_next[~complete, :]
                log_probs = log_probs[~complete]

                if masks is not None:
                    masks = masks[~complete, :]

                # Update VPDA state for remaining sequences
                if guardrails and self.vpda is not None:
                    complete_np = complete.cpu().numpy()
                    self.vpda.stacks = self.vpda.stacks[~complete_np, :]
                    self.vpda.states = self.vpda.states[~complete_np]
                    self.vpda.n_stacks = (~complete_np).sum().item()

                # Stop if all sequences completed (check after removal)
                if idx.size(0) == 0:
                    break

            # Update VPDA for next step if using guardrails
            if guardrails and self.vpda is not None:
                self.vpda.next(idx_next.squeeze())
                next_mask = torch.tensor(self.vpda.get_input_mask()).unsqueeze(1).to(device)
                masks = torch.cat((masks, next_mask), dim=1)

        # Handle any sequences that didn't complete (reached block_size)
        if idx.size(0) > 0:
            for i in range(idx.size(0)):
                completed_rows.append((indexes[i].item(), idx[i], log_probs[i].item()))

        # Sort by original index and separate sequences from log probs
        completed_rows.sort(key=lambda x: x[0])
        sequences = [row[1] for row in completed_rows]
        log_prob_array = np.array([row[2] for row in completed_rows])

        # Pad sequences to same length
        completed_idx = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)

        return completed_idx, log_prob_array
