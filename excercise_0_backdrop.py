import numpy as np


class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # Xavier initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return np.matmul(X, self.W) + self.b

    def backward(self, grad_output):
        self.dW = np.matmul(self.X.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.matmul(grad_output, self.W.T)

        return grad_input


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


# -----------------------------------------------
# ---------------  Test case  -------------------
# -----------------------------------------------

# 1. Generate Data: y = x1 + x2
X_train = np.random.rand(100, 2)
y_true = np.sum(X_train, axis=1, keepdims=True)

# 2. Initialize Network
l1 = LinearLayer(2, 4)
a1 = ReLULayer()
l2 = LinearLayer(4, 1)

learning_rate = 0.01

# 3. Dummy Training Loop
for epoch in range(100):
    # Forward Pass
    z1 = l1.forward(X_train)
    f1 = a1.forward(z1)
    y_pred = l2.forward(f1)

    # Compute Mean Squared Error Loss
    loss = np.mean((y_pred - y_true) ** 2)

    # Backward Pass (Chain Rule)
    # dL/dy_pred for MSE is 2/N * (y_pred - y_true)
    grad = 2.0 * (y_pred - y_true) / y_true.shape[0]

    grad = l2.backward(grad)
    grad = a1.backward(grad)
    grad = l1.backward(grad)

    # Update Weights (SGD)
    l1.W -= learning_rate * l1.dW
    l1.b -= learning_rate * l1.db
    l2.W -= learning_rate * l2.dW
    l2.b -= learning_rate * l2.db

    print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nSample Prediction (Input [0.4, 0.6]):")
# Should eventually approach [[1.0]]
test_val = np.array([[0.4, 0.6]])
print(l2.forward(a1.forward(l1.forward(test_val))))
