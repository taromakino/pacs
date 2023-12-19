import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50


IMG_ENCODE_SIZE = 2048


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet50(weights='IMAGENET1K_V1')
        del self.net.fc
        self.net.fc = nn.Identity()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.net(self.normalize(x))