import numpy as np               # package for numerical linear algebra
import matplotlib.pyplot as plt  # package for plotting and visualization
from math import pi;             # import pi


################################################################################
### Problem 2 ##################################################################

### define equidistant discretization of the interval [0,3*pi/2] with 5 grid points
dom_t = np.linspace(0,pi,150);
print ("grid points:", dom_t);
print ("shape of dom_t:", dom_t.shape);

### define a function fun_g(x) that returns the vector (cos(pi*t),sin(pi*t));
def fun_g(t):
    return [t*np.cos(t*t),t*np.sin(t*t)];

### print the vector of fun_g at t=pi/2
g_pi2 = fun_g(pi/2);
print ("The vector is (%5.4f,%5.4f).\n" % (g_pi2[0],g_pi2[1]));

### evaluate the function g at the grid points of dom_t
[gx,gy] = fun_g(dom_t);
print("values of first coordinate:  ", gx);
print("values of second coordinate: ", gy);

### visualization
plt.figure;

# plot the first and second coordinate separately
plt.subplot(2,1,1);
plt.plot(dom_t, gx, linewidth=2, color=(1,0,0,1));
plt.plot(dom_t, gy, linewidth=2, color=(0,0.9,0,1));
plt.title("t*sin(t*t) and t*cos(t*t)");
plt.xlabel("t");

# plot t -> fun_g(t) 
plt.subplot(2,1,2);
plt.plot(gx, gy, linewidth=2, color=(0,0,1,1));
plt.title("(t*cos(t*t),t*sin(t*t))");
plt.xlabel("g_1");
plt.ylabel("g_2");
plt.axis("equal");

# show the figure
plt.show();


################################################################################

