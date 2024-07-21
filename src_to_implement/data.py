from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode

        if self.mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomRotation(10),
                tv.transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filename']
        label1 = row['crack']
        label2 = row['inactive']

        image = imread(img_path)
        image = gray2rgb(image)
        labels = torch.tensor([float(label1), float(label2)], dtype=torch.float32)

        if self._transform:
            image = self._transform(image)

        return image, labels
