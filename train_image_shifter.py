import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.video_dataset import VideoDataset
from models.conv_autoencoder import ConvAutoencoder
from models.conv_shifter import ConvShifter
from models.u_net import UNetAutoencoder
from models.FSRCNN import FSRCNN
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize
import torchvision.transforms as transforms
from vgg_loss import VGGPerceptualLoss
from torch.cuda import amp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'USING DEVICE: {device}')

EPOCHS = 2000
LEARNING_RATE = 1e-2 # was 1e-3
MSE_LOSS_WEIGHT = 1
VGG_LOSS_WEIGHT = None
BATCH_SIZE = 6
START_SEC= 0
NAME = 'fsrcnn'
MODEL_ITER = None
SAMPLE_SAVE_FREQ = 1000
MODEL_DIR = os.path.expanduser('~/Desktop/Image Shifter/vision_progress')
SAMPLE_VIDEO_PATH = os.path.expanduser('~/Desktop/Image Shifter/dataset/video1.mp4')

if not MSE_LOSS_WEIGHT and not VGG_LOSS_WEIGHT:
    raise ValueError('MUST USE AT LEAST ONE LOSS METRIC')

# Create the dataset and data loader
# img_size = (1920, 1080)
video_dataset = VideoDataset(video_path=SAMPLE_VIDEO_PATH,
                             batch_len_sec=round(BATCH_SIZE / 30 + 1),
                             start_sec=START_SEC,
                             device=device,
                            # transform=torch.nn.Sequential(
                                # Resize((img_size)),  # Resize frames to the target size
                                # Normalize((0.5,), (0.5,), inplace=True)  # Normalize to [-1, 1] range
                            # )
                            )

model = FSRCNN(name=NAME, model_dir=MODEL_DIR).to(device)

sample_count = 0

if MODEL_ITER:
    model.load(MODEL_ITER)
    sample_count = MODEL_ITER


os.makedirs(MODEL_DIR, exist_ok=True)

last_save_point = sample_count

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define the loss function (mean squared error)
if MSE_LOSS_WEIGHT:
    criterion = nn.MSELoss()

print('Start Training')
train_loader = DataLoader(video_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Initialize the VGG perceptual loss model
if VGG_LOSS_WEIGHT:
    vgg_loss_model = VGGPerceptualLoss().to(device)

    # Optionally, define a transform to normalize input to the VGG model
    vgg_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


scaler = amp.GradScaler()

for epoch in range(EPOCHS):
    for batch in train_loader:
        inputs = batch.permute(0, 3, 1, 2).to(device)  # Convert data to (B, C, H, W)
        
        # Zero the gradients
        # optimizer.zero_grad()
        model.zero_grad()


        with amp.autocast():
            # Forward pass
            outputs = model(inputs)
            
            loss = 0

            # Calculate MSE Loss
            if MSE_LOSS_WEIGHT:
                mse_loss = criterion(inputs, outputs)
                loss += MSE_LOSS_WEIGHT * mse_loss
            
            if VGG_LOSS_WEIGHT:
                # Apply VGG normalization to inputs and outputs before passing to VGG model
                inputs_vgg = vgg_transform(inputs)
                outputs_vgg = vgg_transform(outputs)
                
                # Compute perceptual loss
                vgg_loss = nn.functional.mse_loss(vgg_loss_model(inputs_vgg), vgg_loss_model(outputs_vgg))
                loss += VGG_LOSS_WEIGHT * vgg_loss
            
        # Gradient zoom
        scaler.scale(loss).backward()
        # Update generator weight
        scaler.step(optimizer)
        scaler.update()

        # # Backward pass and optimization
        # loss.backward()
        
        # # Clip gradients before optimizer.step()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # optimizer.step()

        sample_count += len(batch)

        # Save the encoder and decoder models every SAMPLE_SAVE_FREQ iterations
        if last_save_point + SAMPLE_SAVE_FREQ <= sample_count:
            model.save(sample_count)
            last_save_point = sample_count

    summary = f'Epoch [{epoch+1}/{EPOCHS}], sample count: {sample_count}, Loss: {loss.item():.4f}'
    if MSE_LOSS_WEIGHT:
        summary += f', MSE Loss: {mse_loss.item():.4f}'
        
    if VGG_LOSS_WEIGHT:
        summary += f', Perceptual Loss: {vgg_loss.item():.4f}'

    print(summary)