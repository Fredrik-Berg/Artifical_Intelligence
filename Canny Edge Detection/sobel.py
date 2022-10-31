#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Fredrik Berg
MatrNr: 12105487
"""

import cv2
import numpy as np


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################

    # Creating the two Sobel-kernels for the x- and y-direction

    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    # Verifying that the values are between 0 and 1 that all values are between 0 and 1

    if img.dtype == np.float32:
        img[img < 0] = 0.
        img[img > 1] = 1.

        # Convolving the image with the two kernels to calculate the approximations of the derivatives

        Gx = cv2.filter2D(img, -1, gx)
        Gy = cv2.filter2D(img, -1, gy)

        # Calculates the gradients and their orientation for every pixel in the image according to formulas in \
        # in the lecture slides on page 109

        gradient = np.sqrt(np.square(Gx) + np.square(Gy))
        orientation = np.arctan2(Gy, Gx)
    else:
        print("It's not the correct datatype!")
        print(img.dtype)

    ######################################################
    return gradient, orientation
