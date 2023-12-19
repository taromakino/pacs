import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.densenet import densenet121


IMG_ENCODE_SIZE = 1024


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = densenet121(weights='IMAGENET1K_V1')
        del self.net.classifier
        self.net.classifier = nn.Identity()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.net(self.normalize(x))