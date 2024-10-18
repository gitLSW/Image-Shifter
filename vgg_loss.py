import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16]  # Extract features up to layer 16
        self.vgg = vgg.eval()
        if not requires_grad:
            for param in self.vgg.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)