import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.regnet import regnet_x_400mf


IMG_ENCODE_SIZE = 400


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = regnet_x_400mf()
        del self.net.fc
        self.net.fc = nn.Identity()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.net(self.normalize(x))