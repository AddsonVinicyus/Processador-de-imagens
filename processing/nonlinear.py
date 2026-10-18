import numpy as np

def median_filter(image, size):
    r = size // 2
    height, width = image.shape

    image_pad = np.pad(image, pad_width=r, mode="edge")

    result = np.zeros((height,width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            neighborhood = image_pad[y:y + size, x:x + size]

            result[y,x] = np.median(neighborhood)

    return result