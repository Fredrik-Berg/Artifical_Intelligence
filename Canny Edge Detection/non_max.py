#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Fredrik Berg
MatrNr: 12105487
"""

import cv2
import numpy as np


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################

    # Fetches necessary variables for iteration, creates the return_value and redefines the orientation so that all of \
    # them are between 0 and pi instead of -pi and pi

    height, width = gradients.shape
    edges = np.zeros([height, width], np.float32)
    pos_ori = orientations
    pos_ori[pos_ori < 0] += np.pi

    # Iterates through the entire image and compares the direct neighbors to thin out edges

    for i in range(1, height-1):
        for j in range(1, width-1):

            # Angle 0, checks if orientation of the gradient is between -22.5 and 22.5 degrees or 157.5 and 202.5 \
            # degrees, if so I check the adjacent pixels in the horizontal direction If then any of the adjacent \
            # pixels are larger than the one we're currently studying we set it to 0, otherwise that pixel is \
            # transferred to the result_array. I chose to keep the current pixel if any of the adjacent ones had a \
            # gradient of equal magnitude.

            if 0 <= pos_ori[i][j] < (1/8 * np.pi) or (7/8 * np.pi) < pos_ori[i][j] <= np.pi:

                p = gradients[i][j+1]
                r = gradients[i][j-1]

                if (gradients[i][j] < p) or (gradients[i][j] < r):
                    edges[i][j] = 0
                else:
                    edges[i][j] = gradients[i][j]

            # Angle 45, checks if orientation of the gradient is between 22.5 and 67.5 degrees or 202.5 and 247.5 \
            # degrees, if so I check the adjacent pixels in the diagonal-right direction If then any of the adjacent \
            # pixels are larger than the one we're currently studying we set it to 0, otherwise that pixel is \
            # transferred to the result_array. I chose to keep the current pixel if any of the adjacent ones had a \
            # gradient of equal magnitude.

            elif (1 / 8 * np.pi) <= pos_ori[i][j] < (3 / 8 * np.pi):

                p = gradients[i+1][j+1]
                r = gradients[i-1][j-1]

                if (gradients[i][j] < p) or (gradients[i][j] < r):
                    edges[i][j] = 0
                else:
                    edges[i][j] = gradients[i][j]

            # Angle 90, checks if orientation of the gradient is between 67.5 and 112.5 degrees or 247.5 and 292.5 \
            # degrees, if so I check the adjacent pixels in the vertical direction If then any of the adjacent \
            # pixels are larger than the one we're currently studying we set it to 0, otherwise that pixel is \
            # transferred to the result_array. I chose to keep the current pixel if any of the adjacent ones had a \
            # gradient of equal magnitude.

            elif (3 / 8 * np.pi) <= pos_ori[i][j] < (5 / 8 * np.pi):

                p = gradients[i+1][j]
                r = gradients[i-1][j]

                if (gradients[i][j] < p) or (gradients[i][j] < r):
                    edges[i][j] = 0
                else:
                    edges[i][j] = gradients[i][j]

            # Angle 135, checks if orientation of the gradient is between 112.5 and 157.5 degrees or 292.5 and 347.5 \
            # degrees, if so I check the adjacent pixels in the diagonal-left direction. If then any of the adjacent \
            # pixels are larger than the one we're currently studying we set it to 0, otherwise that pixel is \
            # transferred to the result_array. I chose to keep the current pixel if any of the adjacent ones had a \
            # gradient of equal magnitude.

            elif (5 / 8 * np.pi) <= pos_ori[i][j] <= (7 / 8 * np.pi):

                p = gradients[i+1][j-1]
                r = gradients[i-1][j+1]

                if (gradients[i][j] < p) or (gradients[i][j] < r):
                    edges[i][j] = 0
                else:
                    edges[i][j] = gradients[i][j]

    ######################################################

    return edges
