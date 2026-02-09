import numpy as np

A = np.array([
    [ 1,  2,  3],
    [-1, -1, -1],
    [ 1,  2,  3]
], dtype=float)

b = np.array([5, 3, -1], dtype=float)

if abs(np.linalg.det(A)) < 1e-10:
    print("A is a singular matrix (no inverse)")
    
    rank_A  = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack((A, b)))

    if rank_A != rank_Ab:
        print("The system of equations has no solution.")
    elif rank_A < A.shape[1]:
        print("The system has infinitely many solutions.")
    else:
        print("The system has a unique solution (rare case)")
else:
    A_inv = np.linalg.inv(A)
    x = A_inv @ b

    print("A^{-1} =")
    print(A_inv)
    print("x =", x)