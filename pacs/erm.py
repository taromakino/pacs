import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from data import N_CLASSES
from encoder_cnn import IMG_ENCODE_SIZE, EncoderCNN
from torch.optim import AdamW
from torchmetrics import Accuracy


class ERM(pl.LightningModule):
    def __init__(self, z_size, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.ecnn = EncoderCNN()
        self.classifier = nn.Sequential(
            nn.Linear(IMG_ENCODE_SIZE, z_size),
            nn.LeakyReLU(),
            nn.Linear(z_size, N_CLASSES)
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_acc = Accuracy('multiclass', num_classes=N_CLASSES)
        self.test_acc = Accuracy('multiclass', num_classes=N_CLASSES)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.ecnn(x).view(batch_size, -1)
        y_pred = self.classifier(x)
        return y_pred, y

    def training_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.cross_entropy(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        y_pred, y = self(*batch)
        if dataloader_idx == 0:
            self.val_acc.update(y_pred, y)
        else:
            assert dataloader_idx == 1
            self.test_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())
        self.log('test_acc', self.test_acc.compute())

    def test_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)