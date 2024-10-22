import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from origami.model import ORIGAMI
from origami.preprocessing import DFDataset, StreamEncoder, target_collate_fn
from origami.utils import FieldToken, Symbol

from .batch_sampler import TargetTokenBatchSampler
from .metrics import Metrics


class Predictor(Metrics):
    def __init__(self, model: ORIGAMI, encoder: StreamEncoder, target_field: str, max_batch_size: int = 128):
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_size = max_batch_size
        self.target_field = target_field
        self.encoder = encoder


    def print_predictions(self, y_true, y_pred):
        print(f"prediction    -->   target")
        for i, (pred, target) in enumerate(zip(y_pred, y_true)):
            pred_dec = self.encoder.decode(pred)
            target_dec = self.encoder.decode(target)

            line_str = f"{i: 5} {pred_dec}   -->   {target_dec}"
            if target == Symbol.UNKNOWN:
                print(line_str)
            else:
                if pred == target:
                    # replace with green version
                    print(f"\033[32m{line_str}\033[0m")
                else:
                    # replace with red version
                    print(f"\033[31m{line_str}\033[0m")


    @torch.no_grad()
    def predict(
        self, data: DFDataset | torch.Tensor, decode: bool = True, show_progress: bool = False
    ) -> list | torch.Tensor:
        """
        Perform prediction on the given dataset using the model, and return the result.

        Args:
            data (DFDataset | Tensor): The dataset to perform prediction on.
            decode (bool, optional): Whether to decode the prediction result. Defaults to True.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            list: The result of the prediction (decoded or integer tokens)
        """

        self.model.eval()
        guardrails = self.model.model_config.guardrails
        device = self.model.device

        # get the target token id so we can chop off the input after the target and only
        # use the sequence up to the target as input
        target_token_id = self.encoder.encode(FieldToken(self.target_field))

        # create sampler to iterate over batches of rows with same target token position
        sampler = TargetTokenBatchSampler(data, target_token_id, self.max_batch_size)

        # setup the dataloader with batch sampler and target collate fn that only returns
        # tokens up to the target field
        loader = DataLoader(data, batch_sampler=sampler, collate_fn=target_collate_fn(target_token_id), num_workers=0)

        # if show_progress is true, wrap loader in a tqdm
        if show_progress:
            loader = tqdm(loader, desc="Predicting")

        predicted = []
        for idx in loader:
            idx = idx.to(device)

            # generate masks if guardrails are used
            if guardrails:
                vpda = self.model.vpda
                vpda.initialize(idx.size(0))
                masks = torch.tensor(vpda.get_sequence_mask(idx)[:, 1:])
                next_mask = torch.tensor(vpda.get_input_mask()).unsqueeze(1)
                masks = torch.cat((masks, next_mask), dim=1)
                masks = masks.to(device)
            else:
                masks = None

            # forward the model, get the logits
            logits, _ = self.model(idx, None, masks)

            # pluck the logits at the final step
            logits = logits[:, -1, :]

            # get the most likely value (arg max)
            idx_next = torch.argmax(logits, dim=-1)

            predicted.extend(i for i in idx_next)

        # switch model back into train mode
        self.model.train()

        # undo sorting operation
        inverse_index = torch.argsort(sampler.sorted_idxs, dim=0, stable=True)
        predicted = torch.tensor(predicted)[inverse_index]

        return self.encoder.decode(predicted) if decode else predicted

    def accuracy(self, dataset: DFDataset, show_progress: bool = False, print_predictions: bool = False) -> float:
        """
        Calculates the accuracy of the model's predictions on the given dataset.

        Parameters:
            dataset (DFDataset): The dataset to calculate the accuracy on.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            float: The accuracy of the model's predictions.
        """

        y_true = self.encoder.encode(dataset.df["target"].to_numpy())
        y_pred = self.predict(dataset, decode=False, show_progress=show_progress).cpu().numpy()

        if print_predictions:
            self.print_predictions(y_true, y_pred)

        return accuracy_score(y_true, y_pred)

    def roc_auc(self, dataset: DFDataset, show_progress: bool = False) -> float:
        """
        Calculates the accuracy of the model's predictions on the given dataset.

        Parameters:
            dataset (DFDataset): The dataset to calculate the accuracy on.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            float: The accuracy of the model's predictions.
        """

        y_pred = self.predict(dataset, decode=False, show_progress=show_progress).cpu().numpy()
        y_true = self.encoder.encode(dataset.df["target"].to_numpy())

        return roc_auc_score(y_true, y_pred)


