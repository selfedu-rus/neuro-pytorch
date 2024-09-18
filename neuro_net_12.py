import torch
from random import randint


def act(z):
    return torch.tanh(z)


def df(z):
    s = act(z)
    return 1 - s * s


def go_forward(x_inp, w1, w2):
    z1 = torch.mv(w1[:, :3], x_inp) + w1[:, 3]
    s = act(z1)

    z2 = torch.dot(w2[:2], s) + w2[2]
    y = act(z2)
    return y, z1, z2


torch.manual_seed(1)

W1 = torch.rand(8).view(2, 4) - 0.5
W2 = torch.rand(3) - 0.5

# обучающая выборка (она же полная выборка)
x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])

lmd = 0.05  # шаг обучения
N = 1000  # число итераций при обучении
total = len(y_train) # размер обучающей выборки

for _ in range(N):
    k = randint(0, total-1)
    x = x_train[k]  # случайный выбор образа из обучающей выборки
    y, z1, out = go_forward(x, W1, W2)  # прямой проход по НС и вычисление выходных значений нейронов
    e = y - y_train[k]  # производная квадратической функции потерь
    delta = e * df(out)  # вычисление локального градиента
    delta2 = W2[:2] * delta * df(z1)  # вектор из 2-х локальных градиентов скрытого слоя

    W2[:2] = W2[:2] - lmd * delta * z1  # корректировка весов связей последнего слоя
    W2[2] = W2[2] - lmd * delta  # корректировка bias

    # корректировка связей первого слоя
    W1[0, :3] = W1[0, :3] - lmd * delta2[0] * x
    W1[1, :3] = W1[1, :3] - lmd * delta2[1] * x

    # корректировка bias
    W1[0, 3] = W1[0, 3] - lmd * delta2[0]
    W1[1, 3] = W1[1, 3] - lmd * delta2[1]

# тестирование обученной НС
for x, d in zip(x_train, y_train):
    y, z1, out = go_forward(x, W1, W2)
    print(f"Выходное значение НС: {y} => {d}")

# результирующие весовые коэффициенты
print(W1)
print(W2)
