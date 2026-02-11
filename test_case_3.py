import numpy as np

matA = np.array([
    [ 1,  2, -3],
    [-1,  1, -1],
    [ 0, -2,  3]
], dtype=float)

matB = None

print("-----Mode-----")
print("1. Gauss Elimination(with pivoting)")
print("2. Gauss Jordan Elimination")
print("3. LU Factorization")
print("4. Inverse Matrix")
n = int(input("Enter Number to choose mode(1-4): "))

print("")
if n in [1, 2, 3]:

    if matB is None:
        raise ValueError("matB is required to solve Ax = b")

    if n == 1:
        from gauss_elimination_with_pivoting import gauss_elimination
        print(gauss_elimination(matA.copy(), matB.flatten()))

    elif n == 2:
        from gauss_jordan_elimination import gauss_jordan
        print(gauss_jordan(matA.copy(), matB.flatten()))

    elif n == 3:
        from LU_facctorization import LU
        print(LU(matA.copy(), matB.copy()))


# ===============================
# CASE 2 : Find Inverse
# ===============================
elif n == 4:
    A = matA.astype(float)
    size = A.shape[0]

    # สร้าง [A | I]
    I = np.eye(size)
    Aug = np.hstack((A, I))

    # Gauss–Jordan Elimination
    for k in range(size):

        # Partial pivoting
        max_row = np.argmax(np.abs(Aug[k:, k])) + k
        if np.isclose(Aug[max_row, k], 0):
            raise ValueError("Matrix is singular")

        Aug[[k, max_row]] = Aug[[max_row, k]]

        # Normalize pivot
        Aug[k] = Aug[k] / Aug[k, k]

        # Eliminate other rows
        for i in range(size):
            if i != k:
                factor = Aug[i, k]
                Aug[i] -= factor * Aug[k]

    invA = Aug[:, size:]

    print("Inverse Matrix:\n", invA)

else:
    raise ValueError("Unknown mode")