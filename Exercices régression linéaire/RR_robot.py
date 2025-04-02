import numpy as np


file_path = '/Users/thomasgibelin/PycharmProjects/DUS4/data/rr_data.csv'
data = np.loadtxt(file_path)
alpha, beta, x_e, y_e = data[0]
A = []
b = []
for alpha, beta, x_e, y_e in data:
    A.append([1.0, 0.0, np.cos(alpha), np.cos(alpha+beta)])
    b.append(x_e)
    A.append([0.0, 1.0, np.sin(alpha), np.sin(alpha+beta)])
    b.append(y_e)
A = np.array(A)
b = np.array(b)

w = np.linalg.pinv(A) @ b
print(w)