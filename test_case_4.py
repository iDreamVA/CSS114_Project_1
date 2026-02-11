import numpy as np

matA = np.array([
    [ 1,  2,  3],
    [-1, -1, -1],
    [ 1,  2,  3]
], dtype=float)
matB = np.array([[5],
                 [-3],
                 [1]], dtype=float)


print("-----Mode-----")
print("1. Gauss Elimination(with pivoting)")
print("2. Gauss Jordan Elimination")
print("3. LU Factorization")
print("4. Inverse Matrix")
n = int(input("Enter Number to choose mode(1-4): "))

print("")

if n == 1:
    from gauss_elimination_with_pivoting import gauss_elimination
    print(gauss_elimination(matA, matB))
elif n == 2:
    from gauss_jordan_elimination import gauss_jordan
    print(gauss_jordan(matA, matB))
elif n == 3:
    from LU_facctorization import LU
    print(LU(matA, matB))
elif n == 4:
    from Inverse_matrix import invMat 
    # check if inverse exists
    if np.isclose(np.linalg.det(matA), 0): 
        raise ValueError("Can't find inverse matrix")
    
    invA = np.linalg.inv(matA)  #find inverse matrix A
    print("Inverse Matrix:\n", invA)
    
    if matB.size != 0: #check size of matrix B
        print(invMat(invA, matB))
else:
    raise ValueError("Unknown mode")