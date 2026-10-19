import numpy as np

def scale_nearest_neighbor(image, scale_factor):
    original_height, original_width = image.shape[:2]

    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    scaled_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            orig_i = min(int(i / scale_factor), original_height - 1)
            orig_j = min(int(j / scale_factor), original_width - 1)

            scaled_image[i, j] = image[orig_i, orig_j]

    return scaled_image
