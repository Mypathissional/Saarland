import numpy as np               # package for numerical linear algebra
import matplotlib.pyplot as plt  # package for plotting and visualization


################################################################################
### Problem 1 ##################################################################

### generate a matrix
A = np.array([[11,12,13,14],
              [21,22,23,24],
              [31,32,33,34]]);

### print the matrix to the screen
print ("A = \n", A);

### print the size of the matrix 
print("Size = ", A.shape)

### set all entries smaller than 24 to -1
A[A<24] = -1

### subtract 30 from the last row
A[-1,:] = A[-1,:]-30

### set the entries of the last column to 0
A[:,-1] = 0
### print the matrix to the screen
print ("A = \n", A);

################################################################################

