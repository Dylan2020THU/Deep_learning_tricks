# Linear Regression with SGD
# 2021-6-8
# TBSI, THU
# ZHX

import numpy as np
from matplotlib import pyplot as plt


def MSE(x, y, training_number):
    '''
    :param x: axis x
    :param y: axis y
    :param training_number: Traning number
    :return: Weight vector & b & the cost of each training
    '''

    # two-dimensional
    w = 0
    b = 0
    alpha = 0.01
    M = len(x)
    y_head = np.zeros(M)

    # training_number = 30
    cost = np.zeros(training_number)
    for num in range(training_number):
        J = 0
        dw = 0
        db = 0
        for i in range(M):
            J += (y_head[i] - y[i]) ** 2  # cost function
            dw += (w * x[i] + b - y[i]) * x[i]
            db += (w * x[i] + b - y[i])
        cost[num] = J / M
        dw = dw / M
        db = db / M

        w = w - alpha * dw
        b = b - alpha * db

        y_head = w * x +b

    return w, b, cost


if __name__ == '__main__':
    np.random.seed(2021)  # set the seed

    M = 10  # The number of data
    # data = np.random.randint(1, 5, [1, M])
    data = np.linspace(5, 40, num=M)  # Generate the dataset
    print('data:', data)

    # Test dataset
    x1 = np.arange(M)
    x2 = data

    # Plot the dataset
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(x1, x2, 'r*')

    w, b, cost = MSE(x1, x2, 20)
    print('w:', w)
    print('b:', b)

    x2_head = w * x1 + b

    ax1.plot(x1, x2_head)

    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(len(cost)), cost)

    plt.show()
