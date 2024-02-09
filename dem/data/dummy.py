import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class DummyDataModule(LightningDataModule):
    def __init__(
        self,
        n_train_batches_per_epoch: int = 100,
        n_val_batches_per_epoch: int = 1,
        n_test_batches_per_epoch: int = 1,
        batch_size: int = 1,
    ):
        super().__init__()
        self.n_train_batches_per_epoch = n_train_batches_per_epoch
        self.n_val_batches_per_epoch = n_val_batches_per_epoch
        self.n_test_batches_per_epoch = n_test_batches_per_epoch
        self.batch_size = batch_size

    def get_dataloader(self, size):
        return DataLoader(np.arange(size * self.batch_size)[:, None], batch_size=self.batch_size)

    def train_dataloader(self):
        return self.get_dataloader(self.n_train_batches_per_epoch)

    def val_dataloader(self):
        return self.get_dataloader(self.n_val_batches_per_epoch)

    def test_dataloader(self):
        return self.get_dataloader(self.n_test_batches_per_epoch)
