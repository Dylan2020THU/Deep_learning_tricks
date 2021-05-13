# Linear Regression with SGD
# 2021-5-13
# TBSI, THU
# ZHX

import numpy as np
from matplotlib import pyplot as plt


np.random.seed(2021)  # set the seed

m = 20  # dimension of data
# data = np.random.randint(1, 5, [1, m])
data = np.linspace(5, 40, num=m)
print('data:', data)

x = np.arange(m)
y = data

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x, y, 'r*')

# two-dimensional
w = 0
b = 0
alpha = 0.01
y_head = np.zeros(m)

training_number = 30
cost = np.zeros(training_number)
# dw = 0
# db = 0
for num in range(training_number):
    J = 0
    dw = 0
    db = 0
    for i in range(m):
        J += (y_head[i] - y[i]) ** 2  # cost function
        dw += (w * x[i] + b - y[i]) * x[i]
        db += (w * x[i] + b - y[i])
    cost[num] = J / m
    dw = dw / m
    db = db / m

    w = w - alpha * dw
    b = b - alpha * db

    y_head = w * x + b
    ax1.plot(x, y_head, label='%s-th'%num)
    plt.legend()

ax2 = fig.add_subplot(122)
ax2.plot(np.arange(len(cost)), cost)

plt.show()
