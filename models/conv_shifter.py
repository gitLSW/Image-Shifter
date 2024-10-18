import os
import torch
import torch.nn as nn
import torch
import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 depthwise=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        if depthwise:
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                padding=padding, groups=in_channels, bias=False
            )
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            )
        else:
            self.depthwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                padding=padding, bias=False
            )
            self.pointwise = None

    def forward(self, x):
        if self.pointwise:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.depthwise(x)
        return x




class ConvShifter(nn.Module):
    def __init__(self, name, model_dir):
        super(ConvShifter, self).__init__()

        self.name = name
        self.model_dir = model_dir

        # Encoder
        self.model = nn.Sequential(
            DepthwiseSeparableConv2d(3, 8, kernel_size=4, stride=2, padding=1),  # (B, 4, H/2, W/2)
            nn.ReLU(),
            DepthwiseSeparableConv2d(8, 16, kernel_size=4, stride=2, padding=1),  # (B, 8, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, output_padding=0),   # (B, 8, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, output_padding=0),    # (B, 3, H, W)
            nn.Sigmoid(),   # Ensure output is in the range [0, 1]
        )

    def forward(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)
    
    def save(self, iteration):
        model_path = os.path.join(self.model_dir, f'{self.name}_model_{iteration}.pth')
        
        torch.save(self.state_dict(), model_path)
        print(f'Saved model state dict at iteration {iteration} to {model_path}')

    def load(self, iteration):
        model_path = os.path.join(self.model_dir, f'{self.name}_model_{iteration}.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.load_state_dict(torch.load(model_path))
        print(f'Loaded model state dict from iteration {iteration} from {model_path}')