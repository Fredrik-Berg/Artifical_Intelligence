import numpy as np
import copy
import open3d as o3d
import camera_params


def fit_plane(pcd: o3d.geometry.PointCloud,
              inlier_threshold: float,
              num_points: int,
              num_iter: int) -> o3d.geometry.PointCloud:

    filtered_pcd = copy.deepcopy(pcd)

    # I use the built-in function segment_plane to remove any points that are a part of the floor

    plane_model, inliers = pcd.segment_plane(distance_threshold=inlier_threshold,
                                             ransac_n=num_points,
                                             num_iterations=num_iter)

    filtered_pcd = o3d.geometry.PointCloud.select_by_index(filtered_pcd, inliers, invert=True)

    return filtered_pcd


def pinhole_projection_rgb(pcd: o3d.geometry.PointCloud) -> np.array:

    # Extract the coefficients of the points in the pointcloud
    object_points = np.asarray(pcd.points)

    # I remove any point where the z-value is equals to 0, both from the pointcloud and my extracted coordinates
    pcd_rgb = pcd.select_by_index(np.where(object_points[:, 2] != 0)[0])
    object_points = object_points[object_points[:, -1] != 0]
    # Extracts the color information of the points in the pointcloud
    object_points_rgb = np.asarray(pcd_rgb.colors)

    # I change the order of the color values so that they are correct for RGB
    object_points_rgb = object_points_rgb[..., ::-1]

    # Creates seperate arrays for the x-, y- and z-values for the points
    x_obj_points = object_points[:, 0]
    y_obj_points = object_points[:, 1]
    z_obj_points = object_points[:, -1]

    # Implementing the formulas given to us regarding pinhole projection
    x_prime = x_obj_points / z_obj_points
    y_prime = y_obj_points / z_obj_points
    u = (camera_params.fx_rgb * x_prime + camera_params.cx_rgb).astype(int)
    v = (camera_params.fy_rgb * y_prime + camera_params.cy_rgb).astype(int)

    # Creates my image-array where I shall store the color values
    proj_image = np.zeros((np.amax(v) + 10, np.amax(u) + 10, 3))

    # Here I place the color-values in the correct place in the image array
    for i in range(object_points_rgb.shape[0]):
        proj_image[v[i]][u[i]][0] = object_points_rgb[i][0]
        proj_image[v[i]][u[i]][1] = object_points_rgb[i][1]
        proj_image[v[i]][u[i]][2] = object_points_rgb[i][2]

    return proj_image


def object_type(place: int) -> str:

    """ In this function I simply check what object we have found so that I can put the text a bit cleaner"""

    id_obj = "hi"
    if place <= 17:
        id_obj = "book"
    if 17 < place <= 39:
        id_obj = "cookiebox"
    if 39 < place <= 56:
        id_obj = "cup"
    if 56 < place <= 78:
        id_obj = "ketchup"
    if 78 < place <= 100:
        id_obj = "sugar"
    if 100 < place <= 122:
        id_obj = "sweets"
    if 122 < place <= 144:
        id_obj = "tea"

    return id_obj


def normalization(matches: np.array, index: np.array, scene: np.array, divider: int, val: int) -> np.array:

    # Creates an array where I will store all my the normalized values for object recognition
    match_mat = np.zeros((7, scene.shape[0]))

    # In this for-loop I divide the found matches of a cluster with the keypoints of the corresponding object. \
    # I then place the calculated value at the correct place in the matrix match_mat.

    for i in range(index.shape[0]):
        cluster_x = index[i]

        if val <= 17:
            match_mat[0][cluster_x] += matches[i] / divider
        if 17 < val <= 39:
            match_mat[1][cluster_x] += matches[i] / divider
        if 39 < val <= 56:
            match_mat[2][cluster_x] += matches[i] / divider
        if 56 < val <= 78:
            match_mat[3][cluster_x] += matches[i] / divider
        if 78 < val <= 100:
            match_mat[4][cluster_x] += matches[i] / divider
        if 100 < val <= 122:
            match_mat[5][cluster_x] += matches[i] / divider
        if 122 < val <= 144:
            match_mat[6][cluster_x] += matches[i] / divider

    return match_mat
