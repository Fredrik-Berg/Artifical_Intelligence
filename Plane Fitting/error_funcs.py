#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Different error functions for plane fitting

Author: Fredrik Berg
MatrNr: 12105487
"""
from typing import Tuple

import numpy as np
import open3d as o3d


def ransac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    """ Calculate the RANSAC error which is the number of outliers.

    The RANSAC error is defined as the number of outliers given a specific model.
    A outlier is defined as a point which distance to the plane is larger (or equal ">=") than a threshold.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param distances: The distances of each point to the proposed plane
    :type distances: np.ndarray with shape (num_of_points,)

    :param threshold: The distance of a point to the plane below which point is considered an inlier (in meters)
    :type threshold: float

    :return: (error, inliers)
        error: The calculated error
        inliers: Boolean mask of inliers (shape: (num_of_points,))
    :rtype: (float, np.ndarray)
    """
    ######################################################

    # Below I find the inliers from the calculated distances and the also compute the error as the number of outliers

    inliers = distances < threshold
    outliers = distances >= threshold
    error = outliers[outliers == True].shape[0]

    ######################################################
    return error, inliers


def msac_error(pcd: o3d.geometry.PointCloud,
               distances: np.ndarray,
               threshold: float) -> Tuple[float, np.ndarray]:
    """ Calculate the MSAC error as defined in https://www.sciencedirect.com/science/article/pii/S1077314299908329

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param distances: The distances of each point to the proposed plane
    :type distances: np.ndarray with shape (num_of_points,)

    :param threshold: The threshold distance at which to change from a quadratic error to a constant error
    :type threshold: float

    :return: (error, inliers)
        error: The calculated error
        inliers: Boolean mask of inliers (shape: (num_of_points,))
    :rtype: (float, np.ndarray)
    """
    ######################################################

    # Below is my implementation for the MSAC-error

    inliers = distances < threshold
    outliers = distances >= threshold
    error = np.sum(inliers * distances ** 2) + np.sum(outliers * threshold**2)

    ######################################################
    return error, inliers


def mlesac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    """ Calculate the MLESAC error as defined in https://www.sciencedirect.com/science/article/pii/S1077314299908329

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param distances: The distances of each point to the proposed plane
    :type distances: np.ndarray with shape (num_of_points,)

    :param threshold: The sigma value needed for MLESAC is calculated as sigma = threshold/2
    :type threshold: float

    :return: (error, inliers)
        error: The calculated error
        inliers: Boolean mask of inliers (shape: (num_of_points,))
    :rtype: (float, np.ndarray)
    """
    ######################################################

    # Here are the initial parameters for my MLESAC-implementation
    gamma = 1/2
    sigma = threshold/2
    v = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    inliers = distances < threshold # I create the boolean array of inliers

    for n in range(1, 4):
        # Calculates p_i, based on equation 19 in the paper and equation for -L in the exercise
        p_i = gamma * (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((distances ** 2) / (2 * (sigma ** 2))))
        # calculates p_o based on equation 20 in the paper
        p_o = (1-gamma) * (1/v)
        # calculates z_i based on equation 18 in the paper
        z_i = p_i/(p_i + p_o)
        # updates gamma based on equation 21 in the paper
        gamma = (1/n) * z_i

    # Calculates -L based on the equation in the exercise description which is our error
    neg_l = -np.sum(np.log(p_i + p_o))
    error = neg_l

    ######################################################

    return error, inliers
