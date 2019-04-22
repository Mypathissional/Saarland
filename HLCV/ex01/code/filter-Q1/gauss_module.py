# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
## additional packages
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.io import imread

def gauss(sigma):
    # ...
    x = np.linspace(-3*sigma, 3*sigma,25)
    Gx = np.exp( -x**2/(2*sigma**2) )/ (sigma * np.sqrt(2*np.pi) )

    return Gx, x

def gaussderiv(img, sigma):
    # ...
    output_image = gaussianfilter(img,sigma)/255.

    [Dx,x] = gaussdx(sigma)
    Dx = np.expand_dims(Dx,0)
    imgDx = convolve2d(output_image,Dx,'same')
    imgDy = convolve2d(output_image,Dx.T,'same')
    
    return imgDx*255., imgDy*255.

def gaussdx(sigma):
    # ...
    x = np.linspace(-3*sigma, 3*sigma,25)
    D = np.exp( -x**2/(2*sigma**2) )/ (np.power(sigma,3) * np.sqrt(2*np.pi) )
    D = np.multiply(D,-x)
    return D, x

def gaussianfilter(img, sigma):

    output_image = img/255.

    kernel = np.expand_dims(gauss(sigma)[0],0)
    output_image = convolve2d(output_image,kernel,'same')
    output_image = convolve2d(output_image, kernel.T,'same')
    return output_image*255.

#img = imread("/home/maria/Documents/uni/classes/hlcv/problems/code/filter-Q1/graf.png")

#gaussianfilter(img, 0.1)
