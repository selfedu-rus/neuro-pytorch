import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from random import randint
import matplotlib.pyplot as plt


class NetGirl(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        return x


model = NetGirl(3, 2, 1)
# print(model)
# print(list(model.parameters()))

# обучающая выборка (она же полная выборка)
x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
total = len(y_train)

optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

model.train()

for _ in range(1000):
    k = randint(0, total-1)
    y = model(x_train[k])
    loss = loss_func(y, y_train[k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

# тестирование обученной НС
for x, d in zip(x_train, y_train):
    y = model(x)
    print(f"Выходное значение НС: {y.data} => {d}")
