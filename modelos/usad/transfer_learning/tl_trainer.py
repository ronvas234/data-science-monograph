from typing import List

import torch
from torch.utils.data import DataLoader

from usad import UsadModel, training


class TransferTrainer:
    """Orchestrates fine-tuning of a USAD-like model. Depends on the abstract
    training() routine from the original repo to keep OCP + DIP.
    """

    def __init__(self, epochs: int, lr: float):
        self.epochs = epochs
        self.lr = lr

    def fit(
        self,
        model: UsadModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[dict]:
        def opt_func(params):
            return torch.optim.Adam(params, lr=self.lr)

        return training(self.epochs, model, train_loader, val_loader, opt_func=opt_func)
