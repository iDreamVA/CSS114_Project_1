import numpy as np

def gauss_jordan(A, b):#The function that solves a system of linear equations Ax = b

    A = A.astype(float) #Convert matrix A to float type
    b = b.astype(float) # Convert vector b to float type
    #To avoid problems with integer division
    n = len(b) #n is the number of equations 

    Aug = np.hstack((A, b.reshape(-1, 1)))# Combine A and b to form an Augmented Matrix [A|b]

    # Gauss–Jordan Elimination

    for k in range(n):# k is the index of the pivot (diagonal)
        pivot_row = np.argmax(np.abs(Aug[k:, k])) + k # Find the row with the largest absolute value of the pivot (partial pivoting)
        if Aug[pivot_row, k] == 0:
            raise ValueError("Matrix is singular")# If the pivot point is 0, it means the matrix is ​​singular.
        
        Aug[[k, pivot_row]] = Aug[[pivot_row, k]]#Swap the pivot row with the row containing the largest pivot.
        for i in range(n):#To make the pivot point equal to 1

            if i != k:# Do not modify the pivot row
                factor = Aug[i, k]# The multiplier used to eliminate values ​​at the pivot column position.
                Aug[i] = Aug[i] - factor * Aug[k] #To make Aug[i, k] become 0
    
    # Aug will be in RREF
    return Aug[:, -1]

print("Gauss-Jordan elimination module loaded.")