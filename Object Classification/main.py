"""
Machine Vision and Cognitive Robotics (376.054)
Exercise 6: Open Challenge
Author: Fredrik Berg (12105487)
"""

from pathlib import Path
from helper_func import *
import numpy as np
import open3d as o3d
from helper_functions_bf import show_image
import cv2
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    # Selects in which image we shall identify the different objects
    img_idx = 4
    training_obj = ["/book000", "/book001", "/book002", "/book003", "/book004", "/book005", "/book006", "/book007",
                    "/book008", "/book009", "/book010", "/book011", "/book012", "/book013", "/book014", "/book015",
                    "/book016", "/cookiebox000", "/cookiebox001", "/cookiebox002", "/cookiebox003", "/cookiebox004",
                    "/cookiebox005", "/cookiebox005", "/cookiebox006", "/cookiebox007", "/cookiebox008",
                    "/cookiebox009", "/cookiebox010", "/cookiebox011", "/cookiebox012", "/cookiebox013",
                    "/cookiebox014", "/cookiebox015", "/cookiebox016", "/cookiebox017", "/cookiebox018",
                    "/cookiebox019", "/cookiebox020", "/cookiebox021", "/cup000", "/cup001", "/cup002", "/cup003",
                    "/cup004", "/cup005", "/cup006", "/cup007", "/cup008", "/cup010", "/cup011", "/cup012", "/cup013",
                    "/cup014", "/cup015", "/cup016", "/ketchup000", "/ketchup001", "/ketchup002", "/ketchup003",
                    "/ketchup004", "/ketchup005", "/ketchup006", "/ketchup007", "/ketchup008", "/ketchup009",
                    "/ketchup010", "/ketchup011", "/ketchup012", "/ketchup013", "/ketchup014", "/ketchup015",
                    "/ketchup016", "/ketchup017", "/ketchup018", "/ketchup019", "/ketchup020", "/ketchup021",
                    "/sugar000", "/sugar001", "/sugar002", "/sugar003", "/sugar004", "/sugar005", "/sugar006",
                    "/sugar007", "/sugar008", "/sugar009", "/sugar010", "/sugar011", "/sugar012", "/sugar013",
                    "/sugar014", "/sugar015", "/sugar016", "/sugar017", "/sugar018", "/sugar019", "/sugar020",
                    "/sugar021", "/sweets000", "/sweets001", "/sweets002", "/sweets003", "/sweets004", "/sweets005",
                    "/sweets006", "/sweets007", "/sweets008", "/sweets009", "/sweets010", "/sweets011", "/sweets012",
                    "/sweets013", "/sweets014", "/sweets015", "/sweets016", "/sweets017", "/sweets018", "/sweets019",
                    "/sweets020", "/sweets021", "/tea000", "/tea001", "/tea002", "/tea003", "/tea004", "/tea005",
                    "/tea006", "/tea007", "/tea008", "/tea009", "/tea010", "/tea011", "/tea012", "/tea013", "/tea014",
                    "/tea015", "/tea016", "/tea017", "/tea018", "/tea019", "/tea020", "/tea021"]

    ####################################################################################################################
    inlier_threshold = 0.01
    test_points = 3
    iterations = 1000
    voxel_size = 0.005
    save_image = False
    matplotlib_plotting = False
    to_normalize = True
    time_array = np.zeros(5)
    ####################################################################################################################

    """Part one of the suggested implementation of the object recognition algorithm. Here we remove the floor/ground
       plane from the pointcloud"""

    # Read Pointcloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/image00")) + str(img_idx) + ".pcd",
                                  remove_nan_points=True,
                                  remove_infinite_points=True)
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Start timer for step 1
    start_time1 = time.time()  # Start the timer

    # Apply plane-fitting algorithm
    filtered_pcd = fit_plane(pcd=pcd, inlier_threshold=inlier_threshold, num_points=test_points, num_iter=iterations)

    # Remove any points left that are a part of the floor from the pointcloud
    _, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=1.5)
    filtered_pcd = filtered_pcd.select_by_index(ind)

    # Calculate elapsed time for step 1
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1
    time_array[0] = elapsed_time1

    # Plot the result of removing the table/floor
    # o3d.visualization.draw_geometries([filtered_pcd])

    ####################################################################################################################
    """Part two of the suggested implementation of the object recognition algorithm. Here we project the remaining 3D-
           points to the 2D-plane"""
    # Start timer for step 2
    start_time2 = time.time()  # Start the timer

    # See the file helper_func for my implementation of pinhole-projection
    proj_image = pinhole_projection_rgb(filtered_pcd)

    # Calculate elapsed time for step 2
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    time_array[1] = elapsed_time2

    # Show projection
    # show_image(proj_image, "Projected 2D-image", save_image=save_image, use_matplotlib=matplotlib_plotting)

    ####################################################################################################################
    """Part three of the suggested implementation of the object recognition algorithm. Here we project the remaining 3D-
       points to the 2D-plane"""

    # Start timer for step 3
    start_time3 = time.time()  # Start the timer

    # I use different values of eps for clustering depending on the image. It is not a good solution to make sure \
    # that I recognize the right amount of clusters for each image.
    eps = 0.0
    if img_idx == 0 or img_idx == 1 or img_idx == 2 or img_idx == 3 or img_idx == 4 or img_idx == 8 or img_idx == 9:
        eps = 0.05
    elif img_idx == 5 or img_idx == 6:
        eps = 0.0375
    else:
        eps = 0.043

    # Uses the built in function cluster_dbscan to identify the different clusters in the images
    cluster_labels = np.array(filtered_pcd.cluster_dbscan(eps=eps, min_points=1000, print_progress=False))

    # Here I color each cluster with a unique color
    final_label = np.amax(cluster_labels)
    colors = plt.get_cmap("tab20")(cluster_labels / final_label)
    colors[cluster_labels < 0] = 0
    cluster_labels = cluster_labels[cluster_labels >= 0]
    col_vals = colors[:, :3]

    filtered_pcd.colors = o3d.utility.Vector3dVector(col_vals)

    # Calculate elapsed time for step 3
    end_time3 = time.time()
    elapsed_time3 = end_time3 - start_time3
    time_array[2] = elapsed_time3

    # Show the resulting pointcloud
    # o3d.visualization.draw_geometries([filtered_pcd])

    ####################################################################################################################
    """Part four of the suggested algorithm, here the clusters are projected from 3D to 2D and where the holes are 
           filled in using one of the morphological transformations"""

    # Start timer for step 4
    start_time4 = time.time()  # Start the timer
    color_values = []
    index = []

    # Here I project the new pointcloud found in step 3 to 2D using my implementation of pinhole projection
    proj_clusters = pinhole_projection_rgb(filtered_pcd)

    # Here I create an index array of the clusters as well as an array to be able to match the color to the \
    # corresponding cluster later on in step 5.

    for i in range(final_label + 1):
        ind = np.argwhere(cluster_labels == i)
        color = colors[ind[0][0], :]
        color_values.append((color[0], color[1], color[2]))
        index.append(i)

    # Changes the lists into arrays for easier use and also changes the order of the color_values for the correct order
    color_values = np.asarray(color_values)
    color_values = color_values[..., ::-1]
    index = np.asarray(index)

    # Here I use the function cv2.morphologyEx to fill in an holes in the projection above
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    image_cluster = cv2.morphologyEx(proj_clusters, cv2.MORPH_CLOSE, kernel)

    # Calculate elapsed time for step 4
    end_time4 = time.time()
    elapsed_time4 = end_time4 - start_time4
    time_array[3] = elapsed_time4

    # Here I show the results of step 4
    #show_image(image_cluster, "Projected Color Cluster Img", save_image=save_image, use_matplotlib=matplotlib_plotting)

    ####################################################################################################################

    """Final part of the suggested algorithm, here I match the different clusters that I've calculated previously
       and match the different objects"""

    ####################################################################################################################
    obj_keypoints = []
    obj_descriptors = []
    match_list = []
    match_obj = np.zeros(index.shape[0])
    best_match = np.zeros(index.shape[0])
    match_images = []
    list_scene_kp = []
    list_col = []
    list_row = []
    sift = cv2.SIFT_create()

    ####################################################################################################################

    # Start timer for step 5
    start_time5 = time.time()  # Start the timer

    # Getting the SIFT keypoints and descriptors for the scene image where we shall recognize the objects
    scene_image = (proj_image * 255.).astype(np.uint8)
    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_image, None)
    pts_kp = cv2.KeyPoint_convert(scene_keypoints)

    # Main loop for loading in all of the objects for recognition
    for i in range(len(training_obj)):
        list_match_kp = []

        # Here I load all of the pointclouds from the training-folder
        pcd_training = o3d.io.read_point_cloud(str(current_path.joinpath("training")) + str(training_obj[i]) + ".pcd",
                                               remove_nan_points=True,
                                               remove_infinite_points=True)

        if not pcd_training.has_points():
            raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

        # Here I project all of the objects from 3D to 2D and typecast them to uint8 for calculation of the SIFT\
        # keypoints and descriptors
        img_obj = pinhole_projection_rgb(pcd_training)
        img_obj = (img_obj * 255.).astype(np.uint8)

        # Getting the SIFT keypoints and descriptors for each object and appending them to their corresponding lists
        object_keypoints, object_descriptors = sift.detectAndCompute(img_obj, None)
        obj_keypoints.append(object_keypoints)
        obj_descriptors.append(object_descriptors)

        # For matching later on
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(object_descriptors, scene_descriptors, k=2)

        # Using lowe's ratio test to remove bad matches
        good_matches = []
        for m in range(len(matches)):
            if matches[m][0].distance < 0.8 * matches[m][1].distance:
                good_matches.append(matches[m][0])

        # Here I take the found matches for each object_image and the scene_image, checks which matches corresponds \
        # to what cluster in the scene_image and adds that value to a list.
        for j in range(len(good_matches)):
            kp = good_matches[j].trainIdx
            col, row = scene_keypoints[kp].pt
            row = round(row)
            col = round(col)
            c = image_cluster[row][col]
            for k in range(color_values.shape[0]):
                if c[0] == color_values[k][0] and c[1] == color_values[k][1] and c[2] == color_values[k][2]:
                    list_match_kp.append(k)

        # Turns the list I found of matches into an array so that I can count the number of matches per cluster easier
        match_key = np.asarray(list_match_kp)
        match_list.append(match_key)

        # Only show every fifth match, otherwise it gets to overwhelming
        match_mask = np.zeros(np.array(matches).shape, dtype=int)
        match_mask[::5, ...] = 1

        draw_params = dict(matchesMask=match_mask,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        matches_img = cv2.drawMatchesKnn(img_obj,
                                         object_keypoints,
                                         scene_image,
                                         scene_keypoints,
                                         matches,
                                         None,
                                         **draw_params)

        # Appends the image with the matches to a list so that one can see them if they want.
        match_images.append(matches_img)

    # Here I find what cluster the keypoints in the scene_image belongs to. The technique is similar to what I did for \
    # the matches. I also finds the coordinates for all the keypoints for text-placement later.

    for i in range(pts_kp.shape[0]):
        col, row = pts_kp[i]
        row = round(row)
        col = round(col)
        c = image_cluster[row][col]
        for j in range(color_values.shape[0]):
            if c[0] == color_values[j][0] and c[1] == color_values[j][1] and c[2] == color_values[j][2]:
                list_scene_kp.append(j)
                list_col.append(col)
                list_row.append(row)
                prev_row = row
                prev_col = col

    # I turn all the found lists in to arrays so that I can decide the number of keypoints per cluster, what \
    # coordinates belong to which cluster more easily.
    scene_key = np.asarray(list_scene_kp)
    norm_scene = np.unique(scene_key, return_counts=True)[1]
    place_row = np.asarray(list_row)
    place_col = np.asarray(list_col)
    placement = np.c_[place_row, place_col, scene_key]

    # If normalization of the match-values is used I enter here to determine the object.
    if to_normalize:

        # I create an array for storing the normalized values here. Later in the method I calculate the normalized \
        # value for each corresponding cluster and object and put them in there corresponding place.
        match_matrix = np.zeros((7, norm_scene.shape[0]))
        for i in range(len(match_list)):
            norm_match = match_list[i]
            norm_index = np.unique(norm_match, return_counts=True)[0]
            norm_count = np.unique(norm_match, return_counts=True)[1]
            val = i
            divider = len(obj_keypoints[i])
            match_mat = normalization(norm_count, norm_index, norm_scene, len(obj_keypoints[i]), val)
            match_matrix = match_matrix + match_mat

        # Here I find which object which had the highest correlation for each cluster
        final_match = np.argmax(match_matrix, axis=0)

        # I then put the text at each clusters median point (found using the placement-array created before) to \
        # state what the object is.
        for i in range(final_match.shape[0]):
            if final_match[i] == 0:
                id_obj = "book"
            if final_match[i] == 1:
                id_obj = "cookiebox"
            if final_match[i] == 2:
                id_obj = "cup"
            if final_match[i] == 3:
                id_obj = "ketchup"
            if final_match[i] == 4:
                id_obj = "sugar"
            if final_match[i] == 5:
                id_obj = "sweets"
            if final_match[i] == 6:
                id_obj = "tea"
            cluster_placement = np.median(placement[placement[:, 2] == i], axis=0)
            cluster_placement = cluster_placement.astype(int)
            coord = np.array([cluster_placement[1], cluster_placement[0]])
            final_image = cv2.putText(proj_image, id_obj, coord, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # If normalization isn't used this matching and plotting method is used. Here I check in each array in my match_list
    # what the value is for each cluster. If the new values is the highest I say that that is the correct match. All \
    # this is stored in the array match_obj.
    else:
        for i in range(len(match_list)):
            norm_match = match_list[i]
            norm_index = np.unique(norm_match, return_counts=True)[0]
            norm_count = np.unique(norm_match, return_counts=True)[1]
            for m in range(norm_index.shape[0]):
                cluster_x = norm_index[m]
                n_count = norm_count[m]
                if n_count > best_match[cluster_x]:
                    best_match[cluster_x] = norm_count[m]
                    match_obj[cluster_x] = i

        match_obj = match_obj.astype(int)

    # Using the array match_obj I find what object corresponds to the value in it. I then use the function \
    # object_type (see helper_func) to identify what object that is and I then put the text at the median of each \
    # cluster.
        for i in range(match_obj.shape[0]):
            id_obj = object_type(match_obj[i])
            cluster_placement = np.median(placement[placement[:, 2] == i], axis=0)
            cluster_placement = cluster_placement.astype(int)
            coord = np.array([cluster_placement[1], cluster_placement[0]])
            final_image = cv2.putText(proj_image, id_obj, coord, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # Calculate elapsed time for step 5
    end_time5 = time.time()
    elapsed_time5 = end_time5 - start_time5
    time_array[4] = elapsed_time5

    # Here I plot the resulting image
    # print(time_array)
    show_image(final_image, "final image", save_image=save_image, use_matplotlib=matplotlib_plotting)
