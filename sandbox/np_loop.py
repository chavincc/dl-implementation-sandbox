import torch

t = torch.tensor(
    [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ]
)

l = t.tolist()
print(l)
