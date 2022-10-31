#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Match descriptors in two images

Author: Fredrik Berg
MatrNr: 12105487
"""
import numpy as np
import cv2


def match_descriptors(descriptors_1: np.ndarray, descriptors_2: np.ndarray, best_only: bool) -> np.ndarray:
    """ Find matches for patch descriptors

    :param descriptors_1: Patch descriptors of first image
    :type descriptors_1: np.ndarray with shape (m1, n) containing m1 descriptors of length n

    :param descriptors_2: Patch descriptor of second image
    :type descriptors_2: np.ndarray with shape (m2, n) containing m2 descriptors of length n

    :param best_only: If True, only keep the best match for each descriptor
    :type best_only: Boolean

    :return: Array representing the successful matches. Each row contains the indices of the matches descriptors
    :rtype: np.ndarray with shape (k, 2) with k being the number of matches
    """
    ######################################################
    # List for creating the array matches
    matches_list = []

    # Here I loop through the descriptors_1-array and then I calculate the euclidian distance for each descriptor \
    # in descriptors_2. I then locate the smallest distance as well.
    for i in range(descriptors_1.shape[0]):
        d = np.sqrt(np.sum(np.square((descriptors_1[i] - descriptors_2)), axis=1))
        location = np.argmin(d)
        # Here I check if best_only is activated to see how many matches between one descriptor in descriptors_2 that \
        # is allowed. My implementation of this wasn't finished so my code sadly only works for best_only == false
        if best_only:
            check = any(location in value for value in matches_list)
            if check:
                checkarray = np.array(matches_list)
                row, col = np.argwhere(checkarray == location)
                d_1 = np.sqrt(np.sum(np.square((descriptors_1[row] - descriptors_2[col])), axis=1))
                d_2 = np.sqrt(np.sum(np.square((descriptors_1[i] - descriptors_2[location])), axis=1))
                if d_1 - d_2 < 0:
                    pass
                else:
                    matches_list.pop([row, col])
                    matches_list.append([i, location])
            else:
                matches_list.append([i, location])
        # Here I just add the match to the list as best_only == False
        else:
            matches_list.append([i, location])

    # Finally I turn the list into an array and return it
    return np.array(matches_list)

    ######################################################
