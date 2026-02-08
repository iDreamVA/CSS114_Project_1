import numpy as np

def gauss_elimination(A, b): #The function that solves a system of linear equations Ax = b

    A = A.astype(float) #Convert matrix A to float type
    b = b.astype(float) # Convert vector b to float type
    #To avoid problems with integer division
    n = len(b) #n is the number of equations 

    #Forward Elimination---Make A the Upper Triangular Matrix.

    for k in range(n-1): # k is the index of the pivot (diagonal position)
        max_row = np.argmax(np.abs(A[k:, k])) + k #Find the row with the largest absolute value in the current column (for pivoting)
        if A[max_row, k] == 0:
            raise ValueError("Matrix is singular")# If the pivot point is 0, the equation cannot be solved
        
        A[[k, max_row]] = A[[max_row, k]]# Swap row k with the row that has the largest pivot in matrix A
        b[[k, max_row]] = b[[max_row, k]]# Swap b as well so that the system of equations remains correct

        for i in range(k+1, n): # Loop to delete the value below the pivot
            factor = A[i, k] / A[k, k] #To make A[i, k] become 0
            A[i, k:] -= factor * A[k, k:] #Remove the row factor (x) from the pivot row at row i
            b[i] -= factor * b[k] #Adjust the value of b to correspond to the change in A

    # Back Substitution---Find x from the equations

    x = np.zeros(n)# Create a solution vector x of size n. Initialize all values ​​to 0

    for i in range(n-1, -1, -1):# Create a solution vector x of size n
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i] # formula to calculate x[i]
    return x