import numpy as np


def Q_values(x, W1, W2, bias_W1, bias_W2):
    # hidden layer
    hidden = np.copy(bias_W1) + np.dot(x, W1)
    hidden = ReLU(hidden)

    # output layer
    output = np.copy(bias_W2) + np.dot(hidden, W2)
    output = ReLU(output)

    return output, hidden


def ReLU(x):
    return np.maximum(0, x)
