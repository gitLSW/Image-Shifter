import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dataset.video_dataset import VideoDataset
from models.u_net import UNetAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NAME = 'unet'
MODEL_ITER = 44_734
MODEL_DIR = os.path.expanduser('~/Desktop/Image Shifter/vision_progress')
SAMPLE_VIDEO_PATH = os.path.expanduser('~/Desktop/Image Shifter/dataset/video1.mp4')

dummy_input = VideoDataset(video_path=SAMPLE_VIDEO_PATH,
                             batch_len_sec=1/20,
                             start_sec=0,
                             device=device,
                            # transform=torch.nn.Sequential(
                                # Resize((img_size)),  # Resize frames to the target size
                                # Normalize((0.5,), (0.5,), inplace=True)  # Normalize to [-1, 1] range
                            # )
                            ).frames[0].unsqueeze(0).permute(0, 3, 1, 2)

# Load the model and set to evaluation mode
model = UNetAutoencoder(name=NAME, model_dir=MODEL_DIR).to(device)
model.load(MODEL_ITER)

model.eval()  # Set the model to evaluation mode

# Export the model
torch.onnx.export(model, dummy_input, f'{MODEL_DIR}/{NAME}.onnx', opset_version=11)