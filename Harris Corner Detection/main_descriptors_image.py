#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision and Cognitive Robotics (376.054)
Exercise 2b: Harris Corners
Clara Haider, Matthias Hirschmanner 2021
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""
from pathlib import Path

from descriptor_functions import *
from harris_corner import harris_corner
from helper_functions import *
from match_descriptors import *

if __name__ == '__main__':
    save_image = False  # Enables saving of matches image
    img_path_1 = 'desk/Image-00.jpg'  # Try different images
    img_path_2 = 'desk/Image-01.jpg'

    # parameters <<< try different settings!
    sigma1 = 0.8
    sigma2 = 1.5
    threshold = 0.01
    k = 0.04
    patch_size = 3

    current_path = Path(__file__).parent
    img_gray_1 = cv2.imread(str(current_path.joinpath(img_path_1)), cv2.IMREAD_GRAYSCALE)
    if img_gray_1 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_1)))

    img_gray_2 = cv2.imread(str(current_path.joinpath(img_path_2)), cv2.IMREAD_GRAYSCALE)
    if img_gray_2 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_2)))

    # Convert images from uint8 with range [0,255] to float32 with range [0,1]
    img_gray_1 = img_gray_1.astype(np.float32) / 255.
    img_gray_2 = img_gray_2.astype(np.float32) / 255.

    # Choose which descriptor to use by indexing into the cell array
    descriptor_func_ind = 0

    # descriptor function names
    descriptor_funcs = [patch_basic, patch_norm, patch_sort, patch_sort_circle, block_orientations]

    # Patch size must be 16 for block_orientations
    if descriptor_func_ind == 4:
        patch_size = 16

    descriptor_func = descriptor_funcs[descriptor_func_ind]

    # Harris corner detector
    _, _, _, _, _, _, _, _, corners = harris_corner(img_gray_1, sigma1=sigma1, sigma2=sigma2, threshold=threshold, k=k)

    # Create descriptors
    interest_points_1, descriptors_1 = compute_descriptors(descriptor_func, img_gray_1, corners[:, 0:2], patch_size)
    print(descriptors_1.shape)
    # Harris corner detector
    _, _, _, _, _, _, _, _, corners = harris_corner(img_gray_2, sigma1=sigma1, sigma2=sigma2, threshold=threshold, k=k)

    # Create descriptors
    interest_points_2, descriptors_2 = compute_descriptors(descriptor_func, img_gray_2, corners[:, 0:2], patch_size)

    # Match descriptors
    matches = match_descriptors(descriptors_1, descriptors_2, best_only=False)

    # Display results
    show_matches(img_gray_1, img_gray_2, interest_points_1, interest_points_2, matches, save_image=save_image)
