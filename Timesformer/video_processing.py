import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

def get_frames(video_path, num_frames = 20):
    """
    Returns 20 equally spaced frames from a video as a tensor (T, C, H, W), normalized.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        if total_frames < num_frames:
            raise ValueError(f"Video too short: {total_frames} frames < {num_frames}")

        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) < num_frames:
            last_frame = frames[-1]
            frames += [last_frame] * (num_frames - len(frames))

        video_tensor = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # T,C,H,W

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        video_tensor = transform(video_tensor)

        return video_tensor
    finally:
        cap.release()

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=20):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames

    
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        frames = get_frames(path, num_frames=self.num_frames)
        return frames, label



