import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        x=F.linear(x,x)

        return x

model = NetGirl(2, 3, 1)

print(model)

gen_p = model.parameters() # возвращает генератор с набором параметров
print(list(model.parameters())) # отображение списка параметров

x_train = torch.FloatTensor([(1, 1), (1,2), (1, 3), (1, 5),
                            (1,7), (1, 9), (2, 3), (2, 4),(2,5)])
y_train = torch.FloatTensor([2, 3, 4, 6, 8, 10, 5, 6,7])
total = len(y_train)

optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()
model.train()

num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    k = randint(0, total - 1)
    y = model(x_train[k])
    y=y.squeeze()
    loss = loss_func(y, y_train[k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch + 1}/{num_epochs}], Потери: {loss.item():.4f}')

    losses.append(loss.item())

model.eval()

for x, d in zip(x_train, y_train):
    y = model(x)
    print(f"Выходное значение НС: {y.data} => {d}")

# Построение функции потерь

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()