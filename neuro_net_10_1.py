import torch
import matplotlib.pyplot as plt

N = 5

x1 = torch.rand(N)
x2 = x1 + torch.randint(1, 10, [N]) / 10
C1 = torch.vstack([x1, x2]).mT

x1 = torch.rand(N)
x2 = x1 - torch.randint(1, 10, [N]) / 10
C2 = torch.vstack([x1, x2]).mT

f = [0, 1]

w = torch.FloatTensor([-0.3, 0.3])
for i in range(N):
    x = C1[:][i]
    y = torch.dot(w, x)
    if y >= 0:
        print("Класс C1")
    else:
        print("Класс C2")

plt.scatter(C1[:, 0], C1[:, 1], s=10, c='red')
plt.scatter(C2[:, 0], C2[:, 1], s=10, c='blue')
plt.plot(f)
plt.grid()
plt.show()
