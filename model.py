import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, padding=0) # in_chaanel = 1 for grayscale, out_channel = 32 features, kernel_sizs=3
    self.bn1 = nn.BatchNorm2d(32)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=0)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=0)
    self.bn3 = nn.BatchNorm2d(128)
    self.fc1 = nn.Linear(128 * 23 * 23, 512) # 200 => 198 => 99 => 97 => 48 => 46 => 23
    self.bn4 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(512, num_classes)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.pool(self.bn1(self.conv1(x)))
    x = self.pool(self.bn2(self.conv2(x)))
    x = self.pool(self.bn3(self.conv3(x)))

    print(x.shape)
    x = x.view(-1, 128 * 23 * 23)

    x = self.dropout(F.relu(self.bn4(self.fc1(x))))
    x = self.fc2(x)

    return x