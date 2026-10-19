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