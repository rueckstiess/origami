import time
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.dataloader import DataLoader, default_collate

from storm_ml.preprocessing import DFDataset
from storm_ml.utils import auto_device

from .vpda import DocumentVPDA


class GPTBase(nn.Module):
    """GPT Base Language Model."""

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def __init__(
        self,
        model_config,
        train_config,
        vpda: Optional[DocumentVPDA] = None,
    ):
        super().__init__()

        # store arguments
        self.model_config = model_config
        self.train_config = train_config
        self.vpda = vpda

        self.optimizer = None
        self.scheduler = None
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        self._device = auto_device(train_config.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.batch_num = 0
        self.batch_time = 0.0
        self.batch_dt = 0.0
        self.epoch_num = 0
        self.loss = 0.0

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = auto_device(device)
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def collate_fn(self, tokens: torch.tensor) -> tuple[torch.tensor, torch.tensor, Optional[torch.tensor]]:
        """This function prepares takes a 2D tensor of tokens of shape (batch_size, seq_len) and
        returns inputs, targets and optionally guardrails masks for the model.
        """

        tokens = default_collate(tokens)

        if self.model_config.guardrails:
            out_masks = torch.tensor(self.vpda.get_sequence_mask(tokens))[:, 1:].to(self.device)
        else:
            out_masks = None

        # targets are the same as input but shifted by one token
        tokens = tokens.to(self.device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()

        return inputs, targets, out_masks

    def train_model(self, dataset: DFDataset, epochs: int = None, batches: int = None):
        self.train()

        assert epochs or batches, "Must specify number of epochs or number of batches"

        loader = DataLoader(
            dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
        )

        # setup the optimizer
        self.optimizer = self.configure_optimizer(self.train_config)

        # setup the learning rate scheduler
        warmup_batches = self.train_config.n_warmup_batches
        total_iters = warmup_batches + (batches or len(loader) * epochs)

        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_batches)

        linear_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.train_config.lr_end_factor,
            total_iters=total_iters,
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, linear_scheduler], milestones=[warmup_batches]
        )

        # if resuming training from a checkpoint, advance scheduler according to num_batches
        for _ in range(self.batch_num):
            self.scheduler.step()

        try:
            # if we are training for a number of epochs
            if epochs:
                for _ in range(epochs):
                    for batch in loader:
                        self.train_batch(batch)

                    # epoch callbacks and stats
                    self.trigger_callbacks("on_epoch_end")
                    self.epoch_num += 1

            # if we are training for a number of batches
            if batches:
                target_batches = self.batch_num + batches
                while True:
                    for batch in loader:
                        self.train_batch(batch)
                        if self.batch_num >= target_batches:
                            return

                    self.trigger_callbacks("on_epoch_end")
                    self.epoch_num += 1

        except KeyboardInterrupt:
            print(f"model training interrupted after {self.epoch_num} epochs ({self.batch_num} batches total).")

    def train_batch(self, batch):
        config = self.train_config

        # forward the model
        _, self.loss = self(*batch)

        # backprop and update the parameters
        self.zero_grad(set_to_none=True)
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.grad_norm_clip)
        self.optimizer.step()
        self.scheduler.step()

        # batch callbacks and stats
        self.trigger_callbacks("on_batch_end")
        self.batch_num += 1
        tnow = time.time()
        self.batch_dt = tnow - self.batch_time
        self.batch_time = tnow

    def save(self, path: str) -> None:
        """saves the trained weights to file, including batch and epoch num."""
        torch.save(self.state_dict() | {"batch_num": self.batch_num, "epoch_num": self.epoch_num}, path)

    def load(self, path: str) -> None:
        """loads trained weights from file. Note: You need to initialise the model
        instance with the exact same configuration and dataset."""
        state_dict = torch.load(path, map_location=self.device)

        # load batch and epoch num if available (backwards compatible with old format)
        self.batch_num = state_dict.pop("batch_num", 0)
        self.epoch_num = state_dict.pop("epoch_num", 0)

        self.load_state_dict(state_dict)
