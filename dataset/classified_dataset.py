import os
import torch
from torchvision.io import read_video
from torch.utils.data import Dataset

class ClassifiedVideoDataset(Dataset):
    def __init__(self, video_path, batch_len_sec, start_sec, true_secs, device, transform=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        self.video_path = video_path
        self.batch_len_sec = batch_len_sec
        self.device = device
        self.transform = transform
        self.start_sec = start_sec
        self.curr_frame_idx = 0
        self.true_secs = true_secs

    def get_batch(self):
        end_sec = self.start_sec + self.batch_len_sec
        frames, _, info = read_video(self.video_path, start_pts=self.start_sec, end_pts=end_sec, pts_unit='sec')
        print(f"Extracted {frames.shape[0]} frames from video {self.video_path} between {self.start_sec}s and {end_sec}s")

        if frames.shape[0] == 0:
            raise RuntimeError('No frames extracted (the end of the video might have been reached)')
    
        frames = frames.float().to(self.device) / 255.0  # Normalize pixel values
        if self.transform:
            frames = self.transform(frames)

        targets = torch.zeros(frames.shape[0])
        for idx in range(frames.shape[0]):
            frame_time = self.start_sec + idx / info['video_fps']
            is_true = False
            for true_range in self.true_secs:
                start, end = true_range
                if start <= frame_time and end <= true_range:
                    targets[idx] = True
                    break

        self.start_sec = end_sec

        return frames, targets

    # def __len__(self):
    #     return self._frames.shape[0]

    # def __getitem__(self, idx):
    #     frame = self._frames[idx]

    #     self.curr_frame_idx += 1
    #     if self.__len__() == self.curr_frame_idx:
    #         self.load_batch()

    #     if self.transform:
    #         frame = self.transform(frame)
    #     return frame, self._targets[idx]