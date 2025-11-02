import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import numpy as np
import re

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith('.png')
        ], key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

        self.classes = ["01-10", "21-30", "11-20", "41-55", "31-40", "56-65", "66-80", "88+"]

    def __len__(self):
        return len(self.image_paths)

    def get_label(self, idx):
        # Define label ranges
        ranges = [
            (0, 2473, 0),
            (2474, 3654, 1),
            (3655, 5177, 2),
            (5178, 6187, 3),
            (6188, 7368, 4),
            (7369, 8167, 5),
            (8168, 8820, 6),
            (8821, 9165, 7),
        ]
        for start, end, label in ranges:
            if start <= idx <= end:
                return label
        raise ValueError(f"Index {idx} out of range for labeling.")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        image = image.resize((200, 200))  # or (200, 200) if you want original
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.tensor(image).unsqueeze(0)  # [1, H, W]

        label = self.get_label(idx)
        return image, torch.tensor(label).long()

dataset = CustomDataset('./images')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

print("Classes:", dataset.classes)
print(f"Train samples: {len(train_dataset)}")
