import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module


def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#
# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain
#       handles to distance and histogram functions, and to find out whether histogram function
#       expects grayvalue or color image
#


def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

    model_hists = compute_histograms(
        model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(
        query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images)))

    # your code here
    for model_image_hist_index in range(D.shape[0]):
        for query_image_hist_index in range(D.shape[1]):
            model_hist = model_hists[model_image_hist_index]
            query_hist = query_hists[query_image_hist_index]
            D[model_image_hist_index][query_image_hist_index] = dist_module.get_dist_by_name(
                model_hist, query_hist, dist_type)

    best_match = np.argmin(D, 0)
    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

    image_hist = []

    # compute histogram for each image and add it at the bottom of image_hist
    # your code here
    for image_name in image_list:
        image = np.array(Image.open(image_name)).astype('double')
        if hist_isgray:
            image = rgb2gray(image)

        image_hist.append(histogram_module.get_hist_by_name(
            image, num_bins, hist_type))

    return image_hist

#
# for each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# note: use the previously implemented function 'find_best_match'
# note: use subplot command to show all the images in the same Python figure, one row per query image
#


def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    num_nearest = 5  # show the top-5 neighbors

    [best_matches, D] = find_best_match(
        model_images, query_images, dist_type, hist_type, num_bins)

    best_matches = np.argsort(D, 0)[:num_nearest, :]

    # your code here
    f, axarr = plt.subplots(len(query_images), num_nearest + 1)

    for i, image in enumerate(query_images):

        neighbors = best_matches[:, i]
        axarr[i, 0].imshow(Image.open(query_images[i]))
        for j in range(len(neighbors)):
            axarr[i, 1 + j].imshow(Image.open(model_images[neighbors[j]]))

    plt.show()
