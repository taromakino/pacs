import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from data import N_CLASSES
from torch.optim import Adam
from torchmetrics import Accuracy
from torchvision.models import resnet50


class ERM(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.resnet = resnet50(weights='IMAGENET1K_V1')
        del self.resnet.fc
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, N_CLASSES)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.val_acc = Accuracy('multiclass', num_classes=N_CLASSES)
        self.test_acc = Accuracy('multiclass', num_classes=N_CLASSES)

    def forward(self, x, y, e):
        y_pred = self.normalize(x)
        y_pred = self.resnet.conv1(y_pred)
        y_pred = self.resnet.bn1(y_pred)
        y_pred = self.resnet.relu(y_pred)
        y_pred = self.resnet.maxpool(y_pred)

        y_pred = self.resnet.layer1(y_pred)
        y_pred = self.resnet.layer2(y_pred)
        y_pred = self.resnet.layer3(y_pred)
        y_pred = self.resnet.layer4(y_pred)

        y_pred = self.resnet.avgpool(y_pred)
        y_pred = torch.flatten(y_pred, 1)

        y_pred = self.dropout(y_pred)
        y_pred = self.fc(y_pred)
        return y_pred, y

    def on_train_start(self):
        for module in self.resnet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(False)
                module.eval()

    def training_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.cross_entropy(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)