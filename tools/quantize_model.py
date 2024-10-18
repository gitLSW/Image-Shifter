import os
import torch
from models.conv_shifter import ConvShifter
from models.u_net import UNetAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NAME = 'unet'
MODEL_ITER = 44_734
MODEL_DIR = os.path.expanduser('~/Desktop/Image Shifter/vision_progress')


# Load the model and set to evaluation mode
model = UNetAutoencoder(name=NAME, model_dir=MODEL_DIR).to(device)
model.load(MODEL_ITER)

# Prepare model for quantization
model.quantize()

print(f'Successfully quantized model {NAME}_model_{MODEL_ITER}')

# Save the quantized model
save_path = f'{MODEL_DIR}/quantized_{NAME}_model_{MODEL_ITER}.pth'
torch.save(model.state_dict(), save_path)

print(f'Saved model to {save_path}')