# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
## additional packages
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.io import imread

def gauss(sigma):
    # ...
    x = np.linspace(-3*sigma, 3*sigma,9)
    Gx = np.exp( -x**2/(2*sigma**2) )/ (sigma * np.sqrt(2*np.pi) )

    return Gx, x

def gaussderiv(img, sigma):
    # ...
    return imgDx, imgDy

def gaussdx(sigma):
    # ...
    return D, x

def gaussianfilter(img, sigma):
    # ...
	kernel = np.reshape(gauss(sigma)[0],(3,3))

	outimage = np.transpose(np.array([ convolve2d(img[:,:,i]/255.,kernel) for i in range(3)]),(1,2,0))*255

	print outimage
	fig, axes = plt.subplots(ncols=2)

	#axes[0].imshow(img, vmin=0, vmax=1)

	axes[1].imshow(outimage, vmin=0, vmax=1)

	plt.show()
    #return outimage

img = imread("/home/maria/Documents/uni/classes/hlcv/problems/code/filter-Q1/graf.png")

gaussianfilter(img,0.1)
