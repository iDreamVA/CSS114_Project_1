import numpy as np

# **********LU factorization method**********
def LU(matA, matB):
    size = matA.shape[1] # matrix size
    matU = matA.copy() # U starts as A
    matL = np.eye(size) # L starts as identity matrix
    
    for i in range(size):
        # declare pivot position
        pivot_row = i 
        max_val = abs(matU[i][i]) 
        
        for r in range(i+1, size):
            # find the largest value in column i for pivot
            if abs(matU[r][i]) > max_val:
                max_val = abs(matU[r][i])
                pivot_row = r
                
        # swap rows if pivot row is different
        if pivot_row != i:
            matU[[i, pivot_row]] = matU[[pivot_row, i]] # swap rows in U
            matB[[i, pivot_row]] = matB[[pivot_row, i]] # swap rows in B
            
            if i > 0:
                # swap previous columns of L
                matL[[i, pivot_row], :i] = matL[[pivot_row, i], :i]
    
        # check zero pivot
        if np.isclose(matU[i][i], 0):
            raise ValueError("Zero pivot encountered")
        
        for j in range(i+1, size):
            m = matU[j][i]/matU[i][i] 
            
            # find U
            for k in range(i, size):
                matU[j][k] -= m * matU[i][k]
                
            # find L
            matL[j][i] = m 
                
    #find y from Ly = B (forward substitution)
    matY = np.zeros((size, 1))
    for i in range(size):
        matY[i][0] = matB[i][0] - sum(matL[i][j]*matY[j][0] for j in range(i))
        
    #find x from  Ux = y ( back substitution )
    matX = np.zeros((size, 1))     
    for i in range(size-1, -1, -1):
        matX[i][0] = (matY[i][0] - sum(matU[i][j]*matX[j][0] for j in range(i+1, size)))/matU[i][i]
        
    print("Matrix L: \n", matL)
    print("Matrix U: \n", matU)
    print("Matrix Y: \n", matY)
    print("Matrix X: \n", matX)
    
    return ", ".join(
        f"x{i+1} = {matX[i][0]:.4f}" for i in range(matX.shape[0]))
