import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def act(x):
    return 0 if x < 0.5 else 1


def go(house, rock, attr):
    X = torch.FloatTensor([house, rock, attr]).to(device)
    Wh = torch.FloatTensor([[0.3, 0.3, 0], [0.4, -0.5, 1]]).to(device)  # матрица 2x3
    Wout = torch.FloatTensor([-1, 1]).to(device)     # вектор 1х2

    Zh = torch.mv(Wh, X)       # вычисляем сумму на входах нейронов скрытого слоя
    print(f"Значения сумм на нейронах скрытого слоя: {Zh}")

    Uh = torch.FloatTensor([act(x) for x in Zh]).to(device)
    print(f"Значения на выходах нейронов скрытого слоя: {Uh}")

    Zout = torch.dot(Wout, Uh)
    Y = act(Zout)
    print(f"Выходное значение НС: {Y}")

    return Y


house = 1
rock = 0
attr = 1

res = go(house, rock, attr)
if res == 1:
    print("Ты мне нравишься")
else:
    print("Созвонимся")
