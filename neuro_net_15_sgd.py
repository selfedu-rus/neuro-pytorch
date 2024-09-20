import torch
import torch.optim as optim

from random import randint
import matplotlib.pyplot as plt


def model(X, w):
    return X @ w


N = 2
w = torch.FloatTensor(N).uniform_(-1e-5, 1e-5)
w.requires_grad_(True)
x = torch.arange(0, 3, 0.1)

y_train = 0.5 * x + 0.2 * torch.sin(2*x) - 3.0
x_train = torch.tensor([[_x ** _n for _n in range(N)] for _x in x])

total = len(x)
lr = torch.tensor([0.1, 0.01])
loss_func = torch.nn.L1Loss()
optimizer = optim.Adam(params=[w], lr=0.01)

for _ in range(1000):
    k = randint(0, total-1)
    y = model(x_train[k], w)
    loss = loss_func(y, y_train[k])

    loss.backward()
    # w.data = w.data - lr * w.grad
    # w.grad.zero_()
    optimizer.step()
    optimizer.zero_grad()

print(w)
predict = model(x_train, w)

plt.plot(x, y_train.numpy())
plt.plot(x, predict.data.numpy())
plt.grid()
plt.show()
