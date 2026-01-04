import torch

t = torch.tensor(([[1, 2, 3, 4], [10, 20, 30, 40]]))
t_unsqueeze_1 = t[:, 0].unsqueeze(0)
print(t_unsqueeze_1)
t_unsqueeze_0 = t[:, 0].unsqueeze(1)
print(t_unsqueeze_0)
