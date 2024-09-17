import torch


def act(x):
    return 0 if x <= 0 else 1


w_hidden = torch.FloatTensor([[1, 1, -1.5], [1, 1, -0.5]])
w_out = torch.FloatTensor([-1, 1, -0.5])

# C1 = [(1,0), (0,1)]
# C2 = [(0,0), (1,1)]
data_x = [0, 0] # входные данные x1, x2
x = torch.FloatTensor(data_x + [1])

z_hidden = torch.matmul(w_hidden, x)
print(z_hidden)
u_hidden = torch.FloatTensor([act(x) for x in z_hidden] + [1])
print(u_hidden)

z_out = torch.dot(w_out, u_hidden)
y = act(z_out)
print(y)
