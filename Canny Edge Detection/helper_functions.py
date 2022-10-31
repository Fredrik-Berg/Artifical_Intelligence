#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Vision and Cognitive Robotics (376.054)
Exercise 1: Canny Edge Detector
Matthias Hirschmanner 2021
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img: np.array, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib.

    :param img: :param img: Input image
    :type img: np.array with shape (height, width) or (height, width, channels)
    
    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool
    
    :param use_matplotlib: If this is set to True, Matplotlib will be used for plotting, OpenCV otherwise
    :type use_matplotlib: bool
    """

    # First check if img is color or grayscale. Raise an exception on a wrong type.
    if len(img.shape) == 3:
        is_color = True
    elif len(img.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            'The image does not have a valid shape. Expected either (height, width) or (height, width, channels)')

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.

    if use_matplotlib:
        plt.figure()
        plt.title(title)
        if is_color:
            # OpenCV uses BGR order while Matplotlib uses RGB. Reverse the the channels to plot the correct colors
            plt.imshow(img[..., ::-1])
        else:
            plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        if is_color:
            png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) * 255.).astype(np.uint8)
        else:
            png_img = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)*255.).astype(np.uint8)
        cv2.imwrite(title.replace(" ", "_") + ".png", png_img)


def plot_row_intensities(img: np.array, row: int, title: str = "Intensities", save_image: bool = False) -> None:
    """ Plots the intensities of one row of an input image

    :param img: Input grayscale image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param row: Index of the row of the image you want to plot
    :type row: int

    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool

    :return: None
    """

    if row >= img.shape[0] or row < 0:
        raise ValueError("Row index would be outside of the image.")

    # Get the values of the specified row
    row_values = img[row, ...]

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(row_values)), row_values)
    ax.set(xlabel='Column', ylabel='Intensity',
           title=title)
    ax.grid()
    if save_image:
        fig.savefig(title.replace(" ", "_")+'.png')
    plt.show()

    # Convert the image to BGR so we can display the line in red which intensities we are using
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    output_img[row, ...] = (0, 0, 1)
    show_image(output_img, "Imrow")


def plot_kernel(kernel: np.array, save_image: bool = False) -> None:
    """ Plot the kernel as a 3D surface plot
    
    :param kernel: The square 2D array to plot
    :type kernel: np.array with shape (width, height). The array needs to be square, so width = height

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool

    :param save_image: If this is set to True, an image will be saved to disc as kernel.png
    :type save_image: bool
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    width = kernel.shape[0]
    x = np.array(list(range(width)) * width).reshape(width, width).T
    y = x.copy().T
    ax.plot_surface(x, y, kernel, cmap='viridis')
    if save_image:
        fig.savefig('kernel.png')
    plt.show()


def add_gaussian_noise(img: np.array, mean: float = 0.0, sigma: float = 0.1) -> np.array:
    """ Applies additive Gaussian noise to the input grayscale image

    :param img: Input grayscale image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param mean: Mean of the Gaussian distribution the noise is drawn from
    :type mean: float

    :param sigma: Standard deviation of the Gaussian distribution the noise is drawn from
    :type sigma: float

    :return: Image with added noise
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    noisy_img = img.copy()
    if noisy_img.dtype != np.float32:
        noisy_img.astype(np.float32)
        if np.max(noisy_img) > 1.0:
            noisy_img = noisy_img/255.

    noise = sigma * np.random.randn(*noisy_img.shape) + mean
    noisy_img += noise
    cv2.normalize(noisy_img, noisy_img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Sometimes values are outside of limits due to rounding errors. Cut those values:
    noisy_img[noisy_img < 0.] = 0.
    noisy_img[noisy_img > 1.] = 1.
    return noisy_img
