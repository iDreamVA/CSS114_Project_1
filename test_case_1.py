import numpy as np

A = np.array([
    [2,  1,  3],
    [4,  3,  5],
    [6,  5,  5]
])

b_test = np.array([1, 1, 3])

A_inv = np.linalg.inv(A)
result = A_inv @ b_test
print("(x1, x2, x3) = ", tuple(np.round(result, 2)))