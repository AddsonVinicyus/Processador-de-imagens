import numpy as np
import matplotlib.pyplot as plt

from processing.spatial import rgb_to_gray

def calc_histogram(image):

    if len(image.shape) != 2:
        image = rgb_to_gray(image)

    histogram = np.zeros(256, dtype=int)

    height, width = image.shape
    total_pixels = height * width

    for i in range(height):
        for j in range(width):
            intensity = image[i, j]
            histogram[intensity] += 1

    return histogram, total_pixels


def calc_histogram_normalized(histogram, total_pixels):
    return histogram / total_pixels


def histogram_equalization(image):

    image = rgb_to_gray(image)

    height, width = image.shape
    MN = height * width
    L = 256

    histogram = calc_histogram(image)[0]
    cdf = np.cumsum(histogram) #soma dos pixels de 0 até k

    lut_table = np.round((L - 1) * cdf / MN).astype(np.uint8)

    equalized_image = lut_table[image]

    return equalized_image, lut_table






