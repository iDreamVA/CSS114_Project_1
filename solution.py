import numpy as np

def solve_linear_system(A, b, mode="auto"):

    if mode == "auto":
        detA = np.linalg.det(A)
        if abs(detA) < 1e-10:
            return "Singular matrix"
        return np.linalg.solve(A, b)

    elif mode == "gauss":
        from gauss_elimination_with_pivoting import gauss_elimination
        return gauss_elimination(A, b)

    elif mode == "jordan":
        from gauss_jordan_elimination import gauss_jordan
        return gauss_jordan(A, b)

    else:
        raise ValueError("Unknown mode")

print("Solver module loaded.")