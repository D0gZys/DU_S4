import numpy as np
import matplotlib.pyplot as plt

def data0() :

    file_path = '/Users/thomasgibelin/PycharmProjects/DUS4/Exercices régression linéaire/data/data_0.csv'
    data = np.loadtxt(file_path)
    data = np.array(data)

    A = [
        [1.0, x]
        for x in data[:, 0]
    ]

    b= data[:, 1]

    w = np.linalg.pinv(A) @ b

    xs = np.linspace(0, 10, 100)

    ys = [w[0] + w[1] * x for x in xs]

    plt.scatter(data[:, 0], data[:, 1], label='Data')  # nuage de points
    plt.plot(xs, ys, label='Linear regression', color="orange", linewidth=4)  #
    plt.legend()
    plt.grid()
    plt.show()
def data1() :

    file_path = '/Users/thomasgibelin/PycharmProjects/DUS4/Exercices régression linéaire/data/data_1.csv'
    data = np.loadtxt(file_path)
    data = np.array(data)

    A = [
        [1.0, x, x ** 2]
        for x in data[:, 0]
    ]

    b= data[:, 1]

    w = np.linalg.pinv(A) @ b

    xs = np.linspace(0, 10, 100)

    ys = [w[0] + w[1] * x + w[2] * x**2 for x in xs]

    plt.scatter(data[:, 0], data[:, 1], label='Data')  # nuage de points
    plt.plot(xs, ys, label='Linear regression', color="orange", linewidth=4)  #
    plt.legend()
    plt.grid()
    plt.show()
def data2() :

    file_path = '/Users/thomasgibelin/PycharmProjects/DUS4/Exercices régression linéaire/data/data_2.csv'
    data = np.loadtxt(file_path)
    data = np.array(data)

    A = [
        [1.0, x, np.cos(2 * x)]
        for x in data[:, 0]
    ]

    b= data[:, 1]

    w = np.linalg.pinv(A) @ b

    xs = np.linspace(0, 10, 100)

    ys = [w[0] + w[1] * x + w[2] * np.cos(2 * x) for x in xs]

    plt.scatter(data[:, 0], data[:, 1], label='Data')  # nuage de points
    plt.plot(xs, ys, label='Linear regression', color="orange", linewidth=4)  #
    plt.legend()
    plt.grid()
    plt.show()
def data3() :

    file_path = '/Users/thomasgibelin/PycharmProjects/DUS4/Exercices régression linéaire/data/data_3.csv'
    data = np.loadtxt(file_path)
    data = np.array(data)

    A = [
        [np.log(x), np.sin(x)]
        for x in data[:, 0]
    ]

    b= data[:, 1]

    w = np.linalg.pinv(A) @ b

    xs = np.linspace(0, 10, 100)

    ys = [w[0] * np.log(x) + w[1] * np.sin(x) for x in xs]

    plt.scatter(data[:, 0], data[:, 1], label='Data')  # nuage de points
    plt.plot(xs, ys, label='Linear regression', color="orange", linewidth=4)  #
    plt.legend()
    plt.grid()
    plt.show()


data0()
data1()
data2()
data3()