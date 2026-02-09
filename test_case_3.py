import numpy as np

A = np.array([
    [ 1,  2, -3],
    [-1,  1, -1],
    [ 0, -2,  3]
], dtype=float)

A_inv = np.linalg.inv(A)

print("A^{-1} =")
print(A_inv)