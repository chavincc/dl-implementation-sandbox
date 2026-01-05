import torch

t = torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]])
t_unsqueeze_1 = t[:, 0].unsqueeze(0)
print(t_unsqueeze_1)
t_unsqueeze_0 = t[:, 0].unsqueeze(1)
print(t_unsqueeze_0)

t2 = torch.tensor([[[0, 1], [2, 3]], [[6, 7], [8, 9]]])
print("t2 shape", t2.shape)
t2_reshape = t2.reshape(-1, 2)
print(t2_reshape, t2_reshape.shape)
