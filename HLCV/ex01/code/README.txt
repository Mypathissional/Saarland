# ------------- Comments

1-c
# 1 operation - convolution of 1-d gaussian wih 1-d gaussian will give again 2-d gaussian
# filter. Since, the variance is high enough, the white circle is quite big 
# 2 and 5 operations are equivallent since convolution is commutative.
# As well as 3 and 4-th operations
# Since 2 and 5 operations are convolving image with gaussian and then with the derivative in y-direction of gaussian which by properties of convolutions
# is equivalent to just applying gaussian blur and looking at the change in y direction. We are getting a sphere in which the first cut with some horizonal hyperplane is white and the second is black. Since the values in the blurred image gets bigger closer to origin, the values  of the derivative of  blurred image in y direction would get larger as y gets smaller.
# The similiar thing happens to 3 and 4-th operation when derivative of the blurred image in x direction is calculated only now the sphere is cut into vertical direction since we look at the change in the x axes
 
1-d as expected it gives the edge outlines of the picture. It can be noticed that looking only in direction is not enough, since there can be edges in other directions. For example, looking in x-direction one would completely ignore the white line above the car while it can be seen in the y direction


