import numpy as np               # package for numerical linear algebra
import matplotlib.pyplot as plt  # package for plotting and visualization


################################################################################
### Problem 3 ##################################################################

### Consider the ODE
#
# x'(t) = A*(x_1(t), x_2(t))
#
# where A is a 2x2 matrix and x = (x_1, x_2). This code plots the vector field
# (the right hand side) on a certain grid of the phase space.
#

### define the grid of the phase space
[dom_X, dom_Y] = np.meshgrid(np.linspace(-2,2,20), np.linspace(-2,2,20));

### define a transformation matrix A (2x2 matrix)
A1 = np.array([[1 ,0],
              [0,1]]);

A2 = np.array([[-1 ,0],
              [0,1]]);

A3 = np.array([[0,1],
              [-1,0]]);

### compute the vector on right hand side of the ODE for each point of the grid
# >>> TODO <<<

### visualiz the vector field
plt.figure;
x = dom_X.flatten()
y = dom_Y.flatten()
[u1,v1] = np.matmul(A1,np.array([x, y]))
[u2,v2] = np.matmul(A2,np.array([x, y]))
[u3,v3] = np.matmul(A3,np.array([x, y]))


# use the command 'plt.quiver' to draw the vector field
plt.quiver(x, y, u1,v1)

plt.axis("equal");


plt.figure();
# use the command 'plt.quiver' to draw the vector field
plt.quiver(x, y, u2,v2)

plt.axis("equal");



plt.figure();
# use the command 'plt.quiver' to draw the vector field
plt.quiver(x, y, u3,v3)

plt.axis("equal");

# show the figure
plt.show();


################################################################################

