from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim


class ModelStyle(nn.Module):
    def __init__(self):
        super().__init__()
        _model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.mf = _model.features
        self.mf.requires_grad_(False)
        self.requires_grad_(False)
        self.mf.eval()
        self.idx_out = (0, 5, 10, 19, 28, 34)
        self.num_style_layers = len(self.idx_out) - 1 # последний слой для контента

    def forward(self, x):
        outputs = []
        for indx, layer in enumerate(self.mf):
            x = layer(x)
            if indx in self.idx_out:
                outputs.append(x.squeeze(0))

        return outputs


def get_content_loss(base_content, target):
    return torch.mean( torch.square(base_content - target) )


def gram_matrix(x):
  channels = x.size(dim=0)
  g = x.view(channels, -1)
  gram = torch.mm(g, g.mT) / g.size(dim=1)
  return gram


def get_style_loss(base_style, gram_target):
    style_weights = [1.0, 0.8, 0.5, 0.3, 0.1]

    _loss = 0
    i = 0
    for base, target in zip(base_style, gram_target):
        gram_style = gram_matrix(base)
        _loss += style_weights[i] * torch.mean(torch.square(gram_style - target))
        i += 1

    return _loss


img = Image.open('img.jpg').convert('RGB')
img_style = Image.open('img_style.jpg').convert('RGB')

# transforms = models.VGG19_Weights.DEFAULT.transforms()
transforms = tfs_v2.Compose([tfs_v2.ToImage(),
                             tfs_v2.ToDtype(torch.float32, scale=True),
                             ])

img = transforms(img).unsqueeze(0)
img_style = transforms(img_style).unsqueeze(0)
img_create = img.clone()
img_create.requires_grad_(True)

model = ModelStyle()
outputs_img = model(img)
outputs_img_style = model(img_style)

gram_matrix_style = [gram_matrix(x) for x in outputs_img_style[:model.num_style_layers]]
content_weight = 1
style_weight = 1000
best_loss = -1
epochs = 100

optimizer = optim.Adam(params=[img_create], lr=0.01)
best_img = img_create.clone()

for _e in range(epochs):
    outputs_img_create = model(img_create)

    loss_content = get_content_loss(outputs_img_create[-1], outputs_img[-1])
    loss_style = get_style_loss(outputs_img_create, gram_matrix_style)
    loss = content_weight * loss_content + style_weight * loss_style

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    img_create.data.clamp_(0, 1)

    if loss < best_loss or best_loss < 0:
      best_loss = loss
      best_img = img_create.clone()

    print(f'Iteration: {_e}, loss: {loss.item(): .4f}')

x = best_img.detach().squeeze()
low, hi = torch.amin(x), torch.amax(x)
x = (x - low) / (hi - low) * 255.0
x = x.permute(1, 2, 0)
x = x.numpy()
x = np.clip(x, 0, 255).astype('uint8')

image = Image.fromarray(x, 'RGB')
image.save("result.jpg")

print(best_loss)
plt.imshow(x)
plt.show()
