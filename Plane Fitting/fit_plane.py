#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: Fredrik Berg
MatrNr: 12105487
"""

from typing import List, Tuple, Callable

import copy

import numpy as np
import open3d as o3d
import time


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float,
              min_sample_distance: float,
              error_func: Callable) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (best_plane, best_inliers, num_iterations, elapsed_time)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
        num_iterations: The number of iterations that were needed for the sample consensus
        elapsed_time: The execution time of the whole function in seconds for comparing different error functions.
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray, int, float)
    """
    ######################################################

    points = np.asarray(pcd.points) # turn pointcloud to np.array
    num_iterations = 0
    best_plane = np.array([0., 0., 1., 0.])
    best_inliers = np.full(points.shape[0], False)
    m = 3  # number of sample points for calculating epsilon^m
    epsilon_m = (m / points.shape[0]) ** m  # Calculates epsilon^m
    eta = (1 - epsilon_m) ** num_iterations  # Initializes the stopping condition
    lowest_error = np.inf
    start_time = time.time()  # Start the timer

    while eta >= (1 - confidence):

        # Here I create a random generator and chooses 3 sampling points at random.
        rng = np.random.default_rng()
        sample_points = rng.choice(points.shape[0], 3)
        samples = points[sample_points]

        # Here I calculate the distance between the different sample points and to make sure that they are
        # a sufficient distance from one another to achieve stability
        point_dist12 = np.sqrt((samples[1][0] - samples[0][0])**2 + (samples[1][1] - samples[0][1])**2 + \
                       (samples[1][2] - samples[0][2])**2)
        point_dist23 = np.sqrt((samples[2][0] - samples[1][0]) ** 2 + (samples[2][1] - samples[1][1]) ** 2 + \
                               (samples[2][2] - samples[1][2]) ** 2)
        point_dist31 = np.sqrt((samples[0][0] - samples[2][0]) ** 2 + (samples[0][1] - samples[2][1]) ** 2 + \
                               (samples[0][2] - samples[2][2]) ** 2)

        # Here I check the distances of my samples, if it is to small I discard them and choose new ones

        if point_dist12 < min_sample_distance or point_dist23 < min_sample_distance or \
                point_dist31 < min_sample_distance:
            pass
        else:

            # I also increase the number of iterations for added robustness

            num_iterations = num_iterations + 1

            # Here I calculate the plane and its constants made up of my samples
            a = (samples[1][1] - samples[0][1]) * (samples[2][2] - samples[0][2]) - \
                (samples[1][2] - samples[0][2]) * (samples[2][1] - samples[0][1])
            b = (samples[1][2] - samples[0][2]) * (samples[2][0] - samples[0][0]) - \
                (samples[1][0] - samples[0][0]) * (samples[2][2] - samples[0][2])
            c = (samples[1][0] - samples[0][0]) * (samples[2][1] - samples[0][1]) - \
                (samples[1][1] - samples[0][1]) * (samples[2][0] - samples[0][0])
            d = -(a*samples[0][0] + b*samples[0][1] + c*samples[0][2])

            # Here I calculate all the distances and then I also determine all of the inliers in the plane

            distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) \
                        / np.sqrt((a ** 2 + b ** 2 + c ** 2))

            error, inliers = error_func(pcd, distances, inlier_threshold)

            # If the error is lower then the previous lowest error I determine this to be my best plane and the
            # best_inliers. I also update my epsilon^m here and see if I fulfill the condition to stop now

            if error < lowest_error:
                lowest_error = error
                best_inliers = inliers
                epsilon_m = (inliers[inliers == True].shape[0] / points.shape[0]) ** 3
                best_plane = np.array([a, b, c, d])

        eta = (1 - epsilon_m) ** num_iterations

    # Here I refine my calculated plane by using np.linalg.lstsq() and my calculated inliers to find the new constants
    # for the refined plane

    new_points = points[np.where(best_inliers == True)[0]]
    rd = np.ones(new_points.shape[0]).reshape(-1)
    rdp = np.linalg.lstsq(new_points, rd)[0]
    a_re, b_re, c_re, d_re = rdp[0], rdp[1], -1., rdp[2]

    # Now I recalculate the distances to update the inliers and the error

    distances_re = np.abs(a_re * new_points[:, 0] + b_re * new_points[:, 1] + c_re * new_points[:, 2] - d_re) \
                   / np.sqrt((a_re ** 2 + b_re ** 2 + c_re ** 2))
    error_re, inliers_re = error_func(pcd, distances_re, inlier_threshold)

    # Finally I check if this is better than my previous best_plane and make sure that the normal is in the positive
    # z-direction

    if error_re < lowest_error:
        if c_re < 0:
            best_plane = np.array([-a_re, -b_re, -c_re, -d_re])
        else:
            best_plane = np.array([a_re, b_re, c_re, d_re])
            best_inliers = inliers_re

    # Finally I calculate the time for the ransac here

    end_time = time.time()
    elapsed_time = end_time - start_time

    return best_plane, best_inliers, num_iterations, elapsed_time


def filter_planes(pcd: o3d.geometry.PointCloud,
                  min_points_prop: float,
                  confidence: float,
                  inlier_threshold: float,
                  min_sample_distance: float,
                  error_func: Callable) -> Tuple[List[np.ndarray],
                                                 List[o3d.geometry.PointCloud],
                                                 o3d.geometry.PointCloud]:
    """ Find multiple planes in the input pointcloud and filter them out.

    Find multiple planes by applying the detect_plane function multiple times. If a plane is found in the pointcloud,
    the inliers of this pointcloud are filtered out and another plane is detected in the remaining pointcloud.
    Stops if a plane is found with a number of inliers < min_points_prop * number of input points.

    :param pcd: The (down-sampled) pointcloud in which to detect planes
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param min_points_prop: The proportion of points of the input pointcloud which have to be inliers of a plane for it
        to qualify as a valid plane.
    :type min_points_prop: float

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers for each plane.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from a plane to be considered an inlier (in meters).
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (plane_eqs, plane_pcds, filtered_pcd)
        plane_eqs is a list of np.arrays each holding the coefficient of a plane equation for one of the planes
        plane_pcd is a list of pointclouds with each holding the inliers of one plane
        filtered_pcd is the remaining pointcloud of all points which are not part of any of the planes
    :rtype: (List[np.ndarray], List[o3d.geometry.PointCloud], o3d.geometry.PointCloud)
    """
    ######################################################

    plane_eqs = []
    plane_pcds = []
    filtered_pcd = copy.deepcopy(pcd)
    num_inliers = np.asarray(pcd.points).shape[0]  # Used for checking the stopping condition
    run = True
    points = np.asarray(filtered_pcd.points)
    stop = min_points_prop * points.shape[0]  # Stopping condition as defined in the exercise

    while run:
        if num_inliers > stop:
            best_plane, best_inliers, num_iterations, _ = fit_plane(pcd=filtered_pcd,
                                                                    confidence=confidence,
                                                                    inlier_threshold=inlier_threshold,
                                                                    min_sample_distance=min_sample_distance,
                                                                    error_func=error_func)

            # Below here I add the found plane to the best_planes-list, extract that plane from the pointcloud
            # and appends those points to the plane_pcds-list

            mask = np.where(best_inliers == True)[0]
            num_inliers = mask.shape[0]
            plane_eqs.append(best_plane)
            points_to_remove = mask.tolist()
            plane_pcds.append(filtered_pcd.select_by_index(points_to_remove, invert=False))
            filtered_pcd = o3d.geometry.PointCloud.select_by_index(filtered_pcd, points_to_remove, invert=True)

        else:
            run = False # Stops the loop

    return plane_eqs, plane_pcds, filtered_pcd
