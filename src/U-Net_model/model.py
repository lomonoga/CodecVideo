import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from math import log10, sqrt

from video.all_def_noise_video import combine_noise_video


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class VideoDataset(Dataset):
    def __init__(self, good_video_dir, transform=None):
        self.good_video_files = sorted(os.listdir(good_video_dir))
        self.transform = transform
        self.good_video_dir = good_video_dir

    def __len__(self):
        return len(self.good_video_files)

    def __getitem__(self, idx):
        good_video_path = os.path.join(self.good_video_dir, self.good_video_files[idx])
        return good_video_path

    def generate_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            noisy_frame = combine_noise_video(frame)
            if self.transform:
                frame = self.transform(frame)
                noisy_frame = self.transform(noisy_frame)
            yield noisy_frame, frame
        cap.release()


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))


def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        print("Model is in training mode:", model.training)
        running_loss = 0.0
        running_psnr = 0.0
        start_time = time.time()

        for batch_paths in dataloader:
            for video_path in batch_paths:
                frames_generator = dataset.generate_frames(video_path)
                for noisy_frames, frames in frames_generator:
                    noisy_frames = torch.from_numpy(noisy_frames.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    frames = torch.from_numpy(frames.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    optimizer.zero_grad()
                    outputs = model(noisy_frames)
                    loss = criterion(outputs, frames)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    # Calculate PSNR for each frame in the batch
                    psnr = calculate_psnr(outputs.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0),
                                          frames.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0))
                    running_psnr += psnr

        epoch_loss = running_loss / len(dataloader)
        epoch_psnr = running_psnr / len(dataloader)
        end_time = time.time()

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.2f}, Time: {end_time - start_time:.2f}s')

    print('Training complete')


if __name__ == "__main__":
    good_video_dir = '../../resources/clean'

    batch_size = 1
    learning_rate = 1e-3
    num_epochs = 25
    dataset = VideoDataset(good_video_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, dataloader, criterion, optimizer, num_epochs)

    torch.save(model.state_dict(), '../../resources/models')
