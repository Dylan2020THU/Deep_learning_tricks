# Implement the gradient descent for logistic regression
# 2021-3-31
# TBSI
# ZHX

import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# def sigmoid(x):
#     fz = []
#     for i in x:
#         fz.append(1/(1 + math.exp(-i)))
#     return fz

def sigmoid(x):
    return 1/(1 + np.exp(-x))



def derivative(training_data, label):
    w = 0
    b = 0

    dw = (y_hat - label) * training_data
    db = y_hat - label

    alpha = 0.1  # learning rate
    w = w - alpha * dw
    b = b - alpha * db


if __name__ == '__main__':

    # create figures
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # ax1 = Axes3D(fig)
    # ax2 = Axes3D(fig)

    ww = np.linspace(-10, 10, 100)  # 0-13, 1000 points inside
    bb = np.linspace(-10, 10, 100)
    w, b = np.meshgrid(ww,bb)

    x = np.random.random(1)  # input data
    y = 1  # label
    # derivative(x, y)

    z = w * x + b
    # print('z:', z)
    y_hat = sigmoid(z)
    # print('y_hat:', y_hat)
    loss_1 = 0.5 * (y_hat - y) * (y_hat - y)  # MSE
    loss_2 = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    ax1.set_xlabel('w')
    ax1.set_ylabel('b')
    ax1.set_zlabel('loss1')
    ax1.plot_surface(w, b, loss_1)  # draw the spatial mesh

    ax2.set_xlabel('w')
    ax2.set_ylabel('b')
    ax2.set_zlabel('loss2')
    ax2.plot_surface(w, b, loss_2)

    plt.show()
