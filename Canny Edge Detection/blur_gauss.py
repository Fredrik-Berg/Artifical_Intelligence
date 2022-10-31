#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Fredrik Berg
MatrNr: 12105487
"""

import cv2
import numpy as np


def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################

    # Generate an x,y-grid for calculations with sides that are 2 * round(3 * sigma) + 1 wide

    x, y = np.mgrid[-round(3*sigma):round(3*sigma) + 1, -round(3*sigma):round(3*sigma) + 1]

    # Calculation of the gaussian matrix is performed below according to formula in lecture 1 p.99

    gaussian_kernel = (1 / (2 * np.pi * (sigma ** 2))) * (np.exp(-(((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))))

    # Generate blurred image by using filter2D-function

    img_blur = cv2.filter2D(img, -1, gaussian_kernel)

    ######################################################
    return img_blur
