import numpy as np
import json
from pathlib import Path

def sigmoid(Z):
    Sigma = 1 / (1 + np.exp(-Z))
    return Sigma

def loss(Sigma_o, y_true):
    loss = 0.5 * (Sigma_o - y_true)**2
    return loss


#Saving the weights
path = Path(__file__).with_name("weights.json")
checkpoint = path if path.exists() else path.with_name("baseline_weights.json")

if checkpoint.exists():
    print("Weights have already been generated")

    with checkpoint.open(encoding="utf-8") as f:
        arrays = json.load(f)
        W1 = np.array(arrays["W1"])
        W2 = np.array(arrays["W2"])
        b1 = np.array(arrays["b1"])
        b2 = np.array(arrays["b2"])

else:
    W1 = np.random.randn(2, 3)
    W2 = np.random.randn(3, 1)

    b1 = np.random.randn(3)
    b2 = np.random.randn(1)

    arrays = {
        "W1": W1.tolist(),
        "W2": W2.tolist(),
        "b1": b1.tolist(),
        "b2": b2.tolist()
    }

    with open(path, "w") as f:
        json.dump(arrays, f)


#Input and output data
row = 1

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])



Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

epochs = 10000

for epoch in range(epochs):
    for row in range(4):

        x = X[row:row + 1]
        y_true = Y[row:row + 1]



        # First Layer, consider b1 is a threshold thus the sign must be negative

        Z_h = x @ W1 - b1
        Sigma_h = sigmoid(Z_h)

        # Second Layer

        Z_o = Sigma_h @ W2 - b2
        Sigma_o = sigmoid(Z_o)


        if epoch % 100 == 0:
            print(f"Given {x} XOR {y_true}")
            print(f"The output is {Sigma_o}")

        L = loss(Sigma_o, y_true)

        #Backpropagation

        alpha = (Sigma_o - y_true) * Sigma_o * (1 - Sigma_o)       # (1, 1)

        dW2 = Sigma_h.T @ alpha                                      # (3, 1)
        db2 = -np.sum(alpha, axis=0)

        beta = (alpha @ W2.T) * Sigma_h * (1 - Sigma_h)             # (1, 3)

        dW1 = x.T @ beta                                          # (2, 3)
        db1 = -np.sum(beta, axis=0)                                # (3,)

        #Updating values

        n = 0.1 # Learning step

        # Weights 2

        W2 = W2 - n * dW2
        b2 = b2 - n * db2

        # Weights 1

        W1 = W1 - n * dW1
        b1 = b1 - n * db1



arrays = {
    "W1": W1.tolist(),
    "W2": W2.tolist(),
    "b1": b1.tolist(),
    "b2": b2.tolist()
}

with open(path, "w") as f:
    json.dump(arrays, f)
