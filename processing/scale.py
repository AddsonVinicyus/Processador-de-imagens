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


def scale_bilinear(image, scale_factor):
    original_height, original_width = image.shape[:2]

    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    scaled_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            orig_x = i / scale_factor
            orig_y = j / scale_factor

            x1 = int(np.floor(orig_x))
            x2 = min(x1 + 1, original_height - 1)
            y1 = int(np.floor(orig_y))
            y2 = min(y1 + 1, original_width - 1)

            dx = orig_x - x1
            dy = orig_y - y1

            top_left = image[x1, y1]
            top_right = image[x1, y2]
            bottom_left = image[x2, y1]
            bottom_right = image[x2, y2]

            top_interp = (1 - dy) * top_left + dy * top_right
            bottom_interp = (1 - dy) * bottom_left + dy * bottom_right

            scaled_image[i, j] = (1 - dx) * top_interp + dx * bottom_interp

    return scaled_image