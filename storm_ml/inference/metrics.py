import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from storm_ml.model.gpt_torch import GPT
from storm_ml.preprocessing import DFDataset


class Metrics:
    def __init__(self, model: GPT, batch_size: int = 128):
        self.model = model
        self.batch_size = batch_size

    @torch.no_grad()
    def ce_loss(self, dataset: DFDataset) -> float:
        total_loss = 0.0
        total_count = 0

        self.model.eval()

        # setup the dataloader
        loader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=self.model.collate_fn,
            batch_size=self.batch_size,
        )

        for batch in loader:
            # forward the model, get the loss
            _, loss = self.model(*batch)
            b, _ = batch[0].size()
            total_loss += loss.item() * b
            total_count += b

        avg_loss = total_loss / total_count

        self.model.train()

        return avg_loss

    def perplexity(self, dataset: DFDataset) -> float:
        return np.exp(self.ce_loss(dataset))
