import os
import json
import torch
import torchvision
import torchvision.transforms.v2 as tfs

transform = tfs.ToPILImage()

mnist_train = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=True)
mnist_test = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=False)

dir_out = 'dataset'
file_format = 'format.json'
train_data = {'dir': "train", 'data': mnist_train}
test_data = {'dir': "test", 'data': mnist_test}

if not os.path.exists(dir_out):
    os.mkdir(dir_out)

    for info in (train_data, test_data):
        os.mkdir(os.path.join(dir_out, info['dir']))

        for i in range(10):
            os.mkdir(os.path.join(dir_out, info['dir'], f"class_{i}"))


for info in (train_data, test_data):
    for i in range(10):
        path = os.path.join(dir_out, info['dir'], f"class_{i}")
        cls = info['data'].data[info['data'].targets == i]

        for n, x in enumerate(cls):
            x = transform(x)
            x.save(os.path.join(path, f"img_{n}.png"), "png")

targets = dict()
for i in range(10):
    targets[f'class_{i}'] = i

fp = open(os.path.join(dir_out, file_format), "w")
json.dump(targets, fp)
fp.close()
