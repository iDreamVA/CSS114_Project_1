import numpy as np

A = np.array([
    [2,  -1, -3,  1],
    [1,   1,  1, -2],
    [3,   2, -3, -4],
    [-1, -4,  1,  1]
])

b_test = np.array([9, 10, 6, 6])

A_inv = np.linalg.inv(A)
result = A_inv @ b_test
print("(x1, x2, x3, x4) = ", tuple(np.round(result, 2)))