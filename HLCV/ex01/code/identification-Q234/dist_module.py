import numpy as np
# 
# compute chi2 distance between x and y
def dist_chi2(x,y):
# your code here
# compute l2 distance between x and y
	return  np.sum(np.square(x-y)/(x+y+np.power(0.1,8)))
def dist_l2(x,y):
# your code here
	return np.sqrt(np.sum(np.square(x-y)))

# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
# your code here
	intersection = np.minimum(x, y)
	return 1- np.true_divide(np.sum(intersection), np.sum(x))


def get_dist_by_name(x, y, dist_name):
	if dist_name == 'chi2':
		return dist_chi2(x,y)
	elif dist_name == 'intersect':
		return dist_intersect(x,y)
	elif dist_name == 'l2':
		return dist_l2(x,y)
	else:
		assert 'unknown distance: %s'%dist_name
  




