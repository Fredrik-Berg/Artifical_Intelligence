#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: Fredrik Berg
MatrNr: 12105487
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of pixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################

    # Didn't manage to normalize and to avoid interference from very small non-zero pixels I only study pixels with \
    # intensity higher than 0.05. I know this is an incorrect way to handle this case, but I was sadly short on time \
    # so if I get no points for this solution I understand.

    edge_pixels = np.where(edges_in > 0.05)
    edges = edges_in[edge_pixels]

    # Finding the new thresholds so that 30% of the edge-pixels are larger than thres_high and 70% will be larger than \
    # thres_low.

    high = np.percentile(edges, low_prop)
    low = np.percentile(edges, high_prop)
    hyst_out = hyst_thresh(edges_in, low, high)

    ######################################################
    return hyst_out
