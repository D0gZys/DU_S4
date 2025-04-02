import numpy as np
import matplotlib.pyplot as plt

################ data ###############

#data_0.csv
file_path0 = '/data/data_0.csv'
data0 = np.loadtxt(file_path0)
data0 = np.array(data0)

#data_1.csv
file_path1 = '/data/data_1.csv'
data1 = np.loadtxt(file_path1)
data1 = np.array(data1)

#data_2.csv
file_path2 = '/data/data_2.csv'
data2 = np.loadtxt(file_path2)
data2 = np.array(data2)

#data_3.csv
file_path3 = '/data/data_3.csv'
data3 = np.loadtxt(file_path3)
data3 = np.array(data3)

################ A ###############

A0 = [
    [1.0, x]
    for x in data0[:, 0]
]

A1 = [
    [1.0, x, x**2]
    for x in data1[:, 0]
]

A2 = [
    [1.0, x, np.cos(2*x)]
    for x in data2[:, 0]
]

A3 = [
    [np.log(x), np.sin(x)]
    for x in data3[:, 0]
]

################ b ###############

b0 = data0[:, 1]
b1 = data1[:, 1]
b2 = data2[:, 1]
b3 = data3[:, 1]

################ w ###############

w0 = np.linalg.pinv(A0) @ b0
w1 = np.linalg.pinv(A1) @ b1
w2 = np.linalg.pinv(A2) @ b2
w3 = np.linalg.pinv(A3) @ b3

################ x ###############

xs = np.linspace(0, 10, 100)

################ y ###############

ys0 = [w0[0] + w0[1] * x for x in xs]
ys1 = [w1[0] + w1[1] * x + w1[2] * x**2 for x in xs]
ys2 = [w2[0] + w2[1] * x + w2[2] * np.cos(2*x) for x in xs]
ys3 = [w3[0] * np.log(x) + w3[1] * np.sin(x) for x in xs]

################ plot ###############

plt.scatter(data3[:, 0], data3[:, 1], label='Data') #nuage de points
plt.plot(xs, ys3, label='Linear regression', color="orange", linewidth=4) #
plt.legend()
plt.grid()
plt.show()
