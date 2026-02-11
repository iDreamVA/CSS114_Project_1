import numpy as np

# **********Inverse Matrix Method**********	
def invMat(invA, matB):
    matX = invA @ matB # calculate x = A^-1 B
    print("X:\n", matX)
    return ", ".join(
        f"x{i+1} = {matX[i][0]:.4f}" for i in range(matX.shape[0]))
