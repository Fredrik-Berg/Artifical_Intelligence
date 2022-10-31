#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Descriptor functions

Author: Fredrik Berg
MatrNr: 12105487
"""
from typing import Callable

import numpy as np
import cv2

from helper_functions import circle_mask


def patch_basic(patch: np.ndarray) -> np.ndarray:
    """ Return the basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################

    # I rearrange the array so that it's shape is the correct one
    return np.concatenate(patch, axis=None)

    ######################################################


def patch_norm(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################

    # Here I reshape the descriptor as intended as well as normalize it
    norm_patch = patch / np.linalg.norm(patch)
    return np.concatenate(norm_patch, axis=None)

    ######################################################


def patch_sort(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################

    # Here I reshape the descriptor as intended as well as normalize and sort it
    norm_patch = patch / np.linalg.norm(patch)
    sort_patch = np.sort(norm_patch)
    return np.concatenate(sort_patch, axis=None)

    ######################################################


def patch_sort_circle(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################


def block_orientations(patch: np.ndarray) -> np.ndarray:
    """ Compute orientation-histogram based descriptor from a patch

    Orientation histograms from 16 4 x 4 blocks of the patch, concatenated in row major order (1 x 128).
    Each orientation histogram should consist of 8 bins in the range [-pi, pi], each bin being weighted by the sum of
    gradient magnitudes of pixel orientations assigned to that bin.

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (16, 16)

    :return: Orientation-histogram based Descriptor
    :rtype: np.ndarray with shape (1, 128)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, 128))  # Replace this line

    ######################################################


def compute_descriptors(descriptor_func: Callable,
                        img: np.ndarray,
                        locations: np.ndarray,
                        patch_size: int) -> (np.ndarray, np.ndarray):
    """ Calculate the given descriptor using descriptor_func on patches of the image, centred on the locations provided

    :param descriptor_func: Descriptor to compute at each location
    :type descriptor_func: function

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param locations: Locations at which to compute the descriptors (n x 2)
    :type locations: np.ndarray with the shape (n x 2). First column is y (row), second is x (column).

    :param patch_size: Value defining the width and height of the patch around each location
        to pass to the descriptor function.
    :type patch_size: int

    :return: (interest_points, descriptors):
        interest_points: k x 2 array containing the image coordinates [y, x] of the corners.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x m matrix containing the patch descriptors.
            Each row vector stores the descriptor vector of the respective corner
            with m being the length of a descriptor.
            The descriptor at index i belongs to the corner at interest_points[i]
            Corners too close to the image boundary to cut out the image patch should not be contained.
    :rtype: (np.ndarray, np.ndarray)
    """
    ######################################################
    # Creating two lists to help with creating the interestpoints- and descriptors-arrays. Also fetching the number \
    # of corners for for-looping.

    amount, coord = locations.shape
    descriptors_list = []
    interest_points_list = []

    # Here i iterate through all of the corners and then I find the corresponding patches that should be used to \
    # calculate the descriptors around the interest points. I only managed to implement it for task 2.2

    for i in range(amount):
        # Fetches the center-point of the patch and uses it to get the upper left corner of my potential patch
        cr = locations[i][0] - 1
        cc = locations[i][1] - 1
        mask = round(patch_size / 2) - 1

        # Here I handle the edge cases where creating a patch will go outside the allowed indices
        if (cr - mask) < 0 or (cr + mask) > img.shape[0] or (cc - mask) < 0 or (cc - mask) > img.shape[1]:
            pass
        else:
            # Here I create the patch and then I process it as well. I then add it and the interest point to their \
            # corresponding list
            region = img[(cr - mask):(cr + mask + 1), (cc - mask):(cc + mask + 1)]
            descriptor = patch_basic(region)
            interest_point = img[cr][cc]
            descriptors_list.append(descriptor)
            interest_points_list.append(interest_point)

    # Here I make the lists into arrays and then I return them
    return np.array(interest_points_list), np.array(descriptors_list)

    ######################################################
