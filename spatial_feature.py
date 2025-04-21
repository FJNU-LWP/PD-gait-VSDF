import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)



        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)



        self.fc = nn.Linear(64 * 16 * 16, 768)


    def forward(self, x):

        batch_size, num_frames, channels, height, width = x.size()
        x = x.reshape(-1, channels, height, width)


        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(batch_size, num_frames, -1)
        x = self.fc(x)

        return x