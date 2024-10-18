import os
from torchvision.io import read_video
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_path, batch_len_sec, start_sec, device, transform=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        self.video_path = video_path
        self.batch_len_sec = batch_len_sec
        self.device = device
        self.transform = transform
        self.start_sec = start_sec
        self.curr_frame_idx = 0
        self.load_batch()

    def load_batch(self):
        end_sec = self.start_sec + self.batch_len_sec
        self.frames, _, _ = read_video(self.video_path, start_pts=self.start_sec, end_pts=end_sec, pts_unit='sec')
        print(f"Extracted {self.frames.shape[0]} frames from video {self.video_path} between {self.start_sec}s and {end_sec}s")

        if self.frames.shape[0] == 0:
            raise RuntimeError('No frames extracted (the end of the video might have been reached)')

        self.frames = self.frames.float().to(self.device) / 255.0  # Normalize pixel values
        self.start_sec = end_sec
        self.curr_frame_idx = 0

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        frame = self.frames[idx]

        self.curr_frame_idx += 1
        if self.__len__() == self.curr_frame_idx:
            self.load_batch()

        if self.transform:
            frame = self.transform(frame)
        return frame