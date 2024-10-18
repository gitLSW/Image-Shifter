import os
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, name, model_dir):
        super(ConvAutoencoder, self).__init__()

        self.name = name
        self.model_dir = model_dir

        # Encoder
        self.encoder = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, H/4, W/4)
            nn.Conv2d(3, 8, kernel_size=8, stride=4, padding=2),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=4, stride=2, padding=1),  # (B, 128, H/8, W/8)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1, output_padding=0),   # (B, 64, H/4*2, W/4*2) -> (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=8, stride=4, padding=2, output_padding=0),    # (B, 3, H/2*2, W/2*2) -> (B, 3, H, W)
            # nn.Upsample(scale_factor=2, mode='nearest'),  # Reverse MaxPool2d -> (B, 64, H/2, W/2)
            nn.Sigmoid(),   # Ensure output is in the range [0, 1]
        )

        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, H/2, W/2)
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, H/4, W/4)
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, H/8, W/8)
        #     nn.ReLU(),
        # )

        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),  # (B, 128, 240*2, 136*2) -> (B, 128, 480, 272)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),   # (B, 64, 480*2, 272*2) -> (B, 64, 960, 544)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0),    # (B, 3, 960*2, 544*2) -> (B, 3, 1920, 1080)
        #     nn.Sigmoid(),  # Ensure output is in the range [0, 1]
        # )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
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