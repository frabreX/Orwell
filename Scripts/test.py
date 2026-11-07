import numpy as np
import json
import os

def sigmoid(Z):
    Sigma = 1 / (1 + np.exp(-Z))
    return Sigma

def loss(Sigma_o, y_true):
    loss = 0.5 * (Sigma_o - y_true)**2
    return loss


#Saving the weights
path = "weights.json"

if os.path.exists(path):
    print("Weights have already been generated")

    with open(path, "r") as f:
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



for row in range(4):

    x = X[row:row + 1]
    y_true = Y[row:row + 1]



    # First Layer, consider b1 is a threshold thus the sign must be negative

    Z_h = x @ W1 - b1
    Sigma_h = sigmoid(Z_h)

    # Second Layer

    Z_o = Sigma_h @ W2 - b2
    Sigma_o = sigmoid(Z_o)


    print(f"Given {x} XOR {y_true}")
    print(f"The output is {Sigma_o}")

