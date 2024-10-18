import os
import torch
import torch.nn as nn
from torch import quantization

# Takes ca. 70ms for inference and prodcues time excellent images
class UNetAutoencoder(nn.Module):
    def __init__(self, name, model_dir):
        super(UNetAutoencoder, self).__init__()

        self.name = name
        self.model_dir = model_dir

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # (B, 8, H/2, W/2)
            nn.ReLU(),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, H/4, W/4)
            nn.ReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, H/8, W/8)
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, H/4, W/4)
            nn.ReLU(),
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 8, kernel_size=4, stride=2, padding=1),  # (B, 8, H/2, W/2), 64 because it gets shown 2 inputs
            nn.ReLU(),
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # (B, 3, H, W), 16 because it gets shown 2 inputs
            nn.Sigmoid(),  # Ensure output is in the range [0, 1]
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # (B, 16, H/2, W/2)
        enc2 = self.encoder2(enc1)  # (B, 32, H/4, W/4)
        enc3 = self.encoder3(enc2)  # (B, 64, H/8, W/8)

        # Decoder with Skip Connections
        dec3 = self.decoder3(enc3)  # (B, 32, H/4, W/4)
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))  # Concatenate along channel dimension (B, 64, H/2, W/2)
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))  # (B, 3, H, W)

        return dec1
    
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


    def quantize(self):
        # Apply fusion for supported layers
        self.encoder1 = quantization.fuse_modules(self.encoder1, [['0', '1']])
        self.encoder2 = quantization.fuse_modules(self.encoder2, [['0', '1']])
        self.encoder3 = quantization.fuse_modules(self.encoder3, [['0', '1']])
        
        # Set qconfig for the model
        self.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Prepare the model for quantization
        # We need to prepare only the parts of the model where quantization is supported
        self.encoder1 = quantization.prepare(self.encoder1, inplace=False)
        self.encoder2 = quantization.prepare(self.encoder2, inplace=False)
        self.encoder3 = quantization.prepare(self.encoder3, inplace=False)
        
        # For decoder layers, we do not apply prepare if they are not supported
        # The decoder layers are still part of the model but not quantized in this pass

        quantization.convert(self, inplace=True)

        return self
