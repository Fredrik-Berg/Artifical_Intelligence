#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Fredrik Berg
MatrNr: 12105487
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################

    # I never managed to get cv2.connectedComponents to work so I chose to implement the algorithm on page 120 in \
    # the lecture notes. Below I've initialized the necessary variables for iteration through the 2D-array.

    height, width = edges_in.shape
    bitwise_img = np.zeros([height, width], dtype=np.float32)
    ridge_row, ridge_col = np.where(high < edges_in)
    edges = []
    q = []
    visited = []

    # Fill in the list for result and the list for the breadth-first-search with the pixels where the intensity is \
    # higher than thres_high.

    for i in range(ridge_row.size):
        edges.append([ridge_row[i], ridge_col[i]])
        q.append([ridge_row[i], ridge_col[i]])

    # In this while-loop I iterate through the adjacent pixels of the pixels in the q-list, if any of them have an \
    # intensity higher than thres_low and haven't been visited in the algorithm then they are added to the q- and \
    # edge-list. When the q-list is empty I set the intensity of all pixels in the edge-list are set to 1 in the \
    # result-array bitwise_img which is returned

    while q:
        rp_r, rp_c = q.pop(0)
        if [rp_r, rp_c] not in visited:
            visited.append([rp_r, rp_c])

        if edges_in[rp_r - 1][rp_c - 1] > low:
            if [rp_r - 1, rp_c - 1] not in visited:
                edges.append([rp_r - 1, rp_c - 1])
                q.append([rp_r - 1, rp_c - 1])
                visited.append([rp_r - 1, rp_c - 1])

        elif edges_in[rp_r - 1][rp_c] > low:
            if [rp_r - 1, rp_c] not in visited:
                edges.append([rp_r - 1, rp_c])
                q.append([rp_r - 1, rp_c])
                visited.append([rp_r - 1, rp_c])

        elif edges_in[rp_r - 1][rp_c + 1] > low:
            if [rp_r - 1, rp_c + 1] not in visited:
                edges.append([rp_r - 1, rp_c + 1])
                q.append([rp_r - 1, rp_c + 1])
                visited.append([rp_r - 1, rp_c + 1])

        elif edges_in[rp_r][rp_c - 1] > low:
            if [rp_r, rp_c - 1] not in visited:
                edges.append([rp_r, rp_c - 1])
                q.append([rp_r, rp_c - 1])
                visited.append([rp_r, rp_c - 1])

        elif edges_in[rp_r][rp_c + 1] > low:
            if [rp_r, rp_c + 1] not in visited:
                edges.append([rp_r, rp_c + 1])
                q.append([rp_r, rp_c + 1])
                visited.append([rp_r, rp_c + 1])

        elif edges_in[rp_r + 1][rp_c - 1] > low:
            if [rp_r + 1, rp_c - 1] not in visited:
                edges.append([rp_r + 1, rp_c - 1])
                q.append([rp_r + 1, rp_c - 1])
                visited.append([rp_r + 1, rp_c - 1])

        elif edges_in[rp_r + 1][rp_c] > low:
            if [rp_r + 1, rp_c] not in visited:
                edges.append([rp_r + 1, rp_c])
                q.append([rp_r + 1, rp_c])
                visited.append([rp_r + 1, rp_c])

        elif edges_in[rp_r + 1][rp_c + 1] > low:
            if [rp_r + 1, rp_c + 1] not in visited:
                edges.append([rp_r + 1, rp_c + 1])
                q.append([rp_r + 1, rp_c + 1])
                visited.append([rp_r + 1, rp_c + 1])

        else:
            if [rp_r - 1, rp_c - 1] not in visited:
                visited.append([rp_r - 1, rp_c - 1])
            if [rp_r - 1, rp_c] not in visited:
                visited.append([rp_r - 1, rp_c])
            if [rp_r - 1, rp_c + 1] not in visited:
                visited.append([rp_r - 1, rp_c + 1])
            if [rp_r, rp_c - 1] not in visited:
                visited.append([rp_r, rp_c - 1])
            if [rp_r, rp_c + 1] not in visited:
                visited.append([rp_r, rp_c + 1])
            if [rp_r + 1, rp_c - 1] not in visited:
                visited.append([rp_r + 1, rp_c - 1])
            if [rp_r + 1, rp_c] not in visited:
                visited.append([rp_r + 1, rp_c])
            if [rp_r + 1, rp_c + 1] not in visited:
                visited.append([rp_r + 1, rp_c + 1])

    while edges:
        edge_r, edge_c = edges.pop(0)
        bitwise_img[edge_r][edge_c] = 1

######################################################
    return bitwise_img
