# Implement the gradient descent for logistic regression
# 2021-3-31
# TBSI
# ZHX

import numpy as np
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))


def deri(x1, x2):
    w1 = 0
    w2 = 0
    b = 0
    z = w1 * x1 + w2 * x2 + b
    y_hat = sigmoid(z)
    loss = - (y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))

    dw1 = (y_hat - y) * x1
    dw2 = (y_hat - y) * x2
    db = y_hat - y

    alpha = 0.1
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b = b- alpha * db


if __name__ == '__main__':
    x1 = np.random.random(1)
    x2 = np.random.random(1)
    deri(x1, x2)