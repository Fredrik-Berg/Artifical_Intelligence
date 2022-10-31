#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Fredrik Berg
MatrNr: 12105487
"""
import numpy as np
import cv2


def harris_corner(img: np.ndarray, sigma1: float, sigma2: float, k: float, threshold: float):
    """ Detect corners using the Harris corner detector

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: (i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, h_dense, h_nonmax, corners):
        i_xx: squared input image filtered with derivative of gaussian in x-direction
        i_yy: squared input image filtered with derivative of gaussian in y-direction
        i_xy: Multiplication of input image filtered with derivative of gaussian in x- and y-direction
        g_xx: i_xx filtered by larger gaussian
        g_yy: i_yy filtered by larger gaussian
        g_xy: i_xy filtered by larger gaussian
        h_dense: Result of harris calculation for every pixel. Array of same size as input image.
            Values normalized to the range (-inf, 1.]
        h_nonmax: Binary mask of non-maxima suppression. Array of same size as input image.
            1 where values are NOT suppressed, 0 where they are.
        corners: n x 3 array containing all detected corners after thresholding and non-maxima suppression.
            Every row vector represents a corner with the elements [y, x, d]
            (d is the result of the harris calculation)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    """

    ######################################################

    # Below I create the two gaussian filters that will be used for this harris corner detector. I also calculate \
    # the derivatives in the x- and y-directions for my first kernel (the one using sigma1)

    kernel_width = 2 * round(3 * sigma1) + 1
    kernel1 = cv2.getGaussianKernel(kernel_width, sigma1)
    kernel2 = cv2.getGaussianKernel(kernel_width, sigma2)
    gauss1 = np.outer(kernel1, kernel1.transpose())
    gauss2 = np.outer(kernel2, kernel2.transpose())
    derivative_x, derivative_y = np.gradient(gauss1)

    # Here I calculate the two derivatives of the image, Ix and Iy as well as Ixx, Ixy and Iyy

    i_x = cv2.filter2D(img, -1, derivative_x)
    i_y = cv2.filter2D(img, -1, derivative_y)
    i_xx = np.square(i_x)
    i_yy = np.square(i_y)
    i_xy = i_x * i_y

    # Calculates the different parts of the M-matrix, Gxx, Gxy and Gyy

    g_xx = cv2.filter2D(i_xx, -1, gauss2)
    g_yy = cv2.filter2D(i_yy, -1, gauss2)
    g_xy = cv2.filter2D(i_xy, -1, gauss2)

    # Here I calculate the determinant and trace of M to finally calculate R. I also normalize it

    deter_m = (g_xx * g_yy) - (g_xy ** 2)
    trace_m = g_xx + g_yy
    r = deter_m - (k * (trace_m ** 2))
    norm_r = r / np.amax(r)

    # Finally I calculate Harris feature value for each of the pixels in the original image! I also create \
    # a list for the corners array. I then iterate through my out from the non-max and for the values which \
    # are larger than the threshold and if non_max is 1 at that position I add it to the corner-list which \
    # I then turn into the array corners which is returned

    h_nonmax = non_max(norm_r)
    corner_list = []
    corner_row, corner_col = np.where(norm_r > threshold)
    for i in range(corner_row.size):
        if h_nonmax[corner_row[i]][corner_col[i]] == 1:
            corner_list.append([corner_row[i], corner_col[i], h_nonmax[corner_row[i]][corner_col[i]]])

    corners = np.array(corner_list, dtype=int)

    return i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, norm_r, h_nonmax, corners

    ######################################################


def non_max(mat: np.ndarray):
    """ Return a matrix in which only local maxima of the input matrix are set to 1, all other values are set to 0

    :param mat: Input matrix
    :type mat: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range (-inf, 1.]

    :return: Binary Matrix with the same dimensions as the input matrix
    :rtype: np.ndarray with shape (height, width) with dtype = int and values in {0,1}

    """

    ######################################################

    # First i generate 8 matrices where I've shifted the pixels so that the pixel I want to check is in the middle \
    # compared to the original matrix mat. For the rows and columns that get rolled around I set them to nan to avoid \
    # issues.

    # Matrix where the pixel to the right is in the middle
    mat_xp = np.roll(mat, 1, axis=1)
    mat_xp[:, 0] = np.nan
    # Matrix where the pixel to the left is in the middle
    mat_xm = np.roll(mat, -1, axis=1)
    mat_xm[:, -1] = np.nan
    # Matrix where the pixel above is in the middle
    mat_yp = np.roll(mat, 1, axis=0)
    mat_yp[0, :] = np.nan
    # Matrix where the pixel below is in the middle
    mat_ym = np.roll(mat, -1, axis=0)
    mat_ym[-1, :] = np.nan
    # Matrix where the pixel in the lower right corner is in the middle
    mat_dig_p = np.roll(np.roll(mat, -1, axis=0), -1, axis=1)
    mat_dig_p[-1, :] = np.nan
    mat_dig_p[:, -1] = np.nan
    # Matrix where the pixel in the upper left corner is in the middle
    mat_dig_m = np.roll(np.roll(mat, 1, axis=0), 1, axis=1)
    mat_dig_m[0, :] = np.nan
    mat_dig_m[:, 0] = np.nan
    # Matrix where the pixel in the lower left corner is in the middle
    mat_adig_p = np.roll(np.roll(mat, -1, axis=0), 1, axis=1)
    mat_adig_p[-1, :] = np.nan
    mat_adig_p[0, :] = np.nan
    # Matrix where the pixel in the upper right corner is in the middle
    mat_adig_m = np.roll(np.roll(mat, 1, axis=0), -1, axis=1)
    mat_adig_m[0, :] = np.nan
    mat_adig_m[:, -1] = np.nan

    # Here i generate 8 different boolean matrices where the comparisons for all adjacent neighbors of the center \
    # pixel is carried out.
    comp_xp = mat > mat_xp
    comp_yp = mat > mat_yp
    comp_dig_p = mat > mat_dig_p
    comp_adig_p = mat > mat_adig_p
    comp_xm = mat > mat_xm
    comp_ym = mat > mat_ym
    comp_dig_m = mat > mat_dig_m
    comp_adig_m = mat > mat_adig_m

    # Here I generate a complete boolean map which checks where the center pixel is larger than all neighbors
    local_max = comp_xp & comp_yp & comp_dig_p & comp_adig_p & comp_xm & comp_ym & comp_dig_m & comp_adig_m

    # Finally i create the output array where all local maxima are set to 1 and the other pixels set to 0.
    output = np.where(local_max == True, 1, 0)
    return output

    ######################################################
