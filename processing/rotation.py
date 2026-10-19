import numpy as np

def rotation_nearest_neighbor(image, angle):
    height, width = image.shape[:2]
    angle_rad = np.deg2rad(angle)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    points = np.array([
        [-cx, -cy],
        [width - 1 - cx, -cy],
        [width - 1 - cx, height - 1 - cy],
        [-cx, height - 1 - cy]  
    ], dtype=float)

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    points_rot = points @ rotation_matrix.T

    min_x = points_rot[:, 0].min()
    max_x = points_rot[:, 0].max()
    min_y = points_rot[:, 1].min()
    max_y = points_rot[:, 1].max()

    out_width = int(np.ceil(max_x - min_x)) + 1
    out_height = int(np.ceil(max_y - min_y)) + 1

    img_out = np.zeros((out_height, out_width) + image.shape[2:], dtype=image.dtype)

    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)

    for l in range(out_height):
        for c in range(out_width):
            x = min_x + c
            y = min_y + l

            x_rot = ca * x + sa * y + cx
            y_rot = -sa * x + ca * y + cy

            x_src = int(np.floor(x_rot + 0.5))
            y_src = int(np.floor(y_rot + 0.5))

            if 0 <= x_src < width and 0 <= y_src < height:
                img_out[l, c] = image[y_src, x_src]

    return img_out


def rotation_bilinear(image, angle):
    height, width = image.shape[:2]
    angle_rad = np.deg2rad(angle)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    points = np.array([
        [-cx, -cy],
        [width - 1 - cx, -cy],
        [width - 1 - cx, height - 1 - cy],
        [-cx, height - 1 - cy]  
    ], dtype=float)

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    points_rot = points @ rotation_matrix.T

    min_x = points_rot[:, 0].min()
    max_x = points_rot[:, 0].max()
    min_y = points_rot[:, 1].min()
    max_y = points_rot[:, 1].max()

    out_width = int(np.ceil(max_x - min_x)) + 1
    out_height = int(np.ceil(max_y - min_y)) + 1

    img_out = np.zeros((out_height, out_width) + image.shape[2:], dtype=image.dtype)

    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)

    for l in range(out_height):
        for c in range(out_width):
            x = min_x + c
            y = min_y + l

            x_rot = ca * x + sa * y + cx
            y_rot = -sa * x + ca * y + cy

            x_src_floor = int(np.floor(x_rot))
            y_src_floor = int(np.floor(y_rot))
            x_src_ceil = x_src_floor + 1
            y_src_ceil = y_src_floor + 1

            if (0 <= x_src_floor < width and 0 <= y_src_floor < height and
                0 <= x_src_ceil < width and 0 <= y_src_ceil < height):

                dx = x_rot - x_src_floor
                dy = y_rot - y_src_floor

                top_left = image[y_src_floor, x_src_floor]
                top_right = image[y_src_floor, x_src_ceil]
                bottom_left = image[y_src_ceil, x_src_floor]
                bottom_right = image[y_src_ceil, x_src_ceil]
                img_out[l, c] = (1 - dx) * (1 - dy) * top_left + dx * (1 - dy) * top_right + (1 - dx) * dy * bottom_left + dx * dy * bottom_right

    return img_out
