import numpy as np

dim = 5
batch_size = 2

a = np.random.randn(batch_size, dim)
b = np.ones((1, dim))

print("a", a)
print("b", b)

print("a + b", a + b)
