import numpy as np
import matplotlib.pyplot as plt

from processing.spatial import rgb_to_gray

def calc_histogram(image):

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





