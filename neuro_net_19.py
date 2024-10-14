import os
import json

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs


class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(self):
        return self.length


to_tensor = tfs.ToImage()  # PILToTensor
d_train = DigitDataset("dataset", transform=to_tensor)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

it = iter(train_data)
x, y = next(it)
print(len(d_train))
