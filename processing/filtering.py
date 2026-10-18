import numpy as np

def convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    assert kernel_height == kernel_width and kernel_height % 2 == 1, "O filtro dever ser quadrado com dimensões ímpares"

    a = kernel_height // 2 # raio(a) do filtro

    convolution_kernel = np.rot90(kernel, 2) # rotaciona o kernel 180 graus

    padded_image = np.pad(image, pad_width = a, mode='constant', constant_values=0)

    convolved_image = np.zeros_like(image, dtype=float)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved_value = np.sum(region * convolution_kernel)
            convolved_image[i, j] = convolved_value

    #convolved_image = np.clip(convolved_image, 0, 255)

    return convolved_image

def normalize_for_display(image):
    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (image - min_value) / (max_value - min_value)

    return np.round(normalized * 255).astype(np.uint8)

def smoothing_average(size):
    kernel = np.ones((size, size), dtype=np.float32)
    kernel = kernel / (size * size)

    return kernel

def smoothing_weighted_average(size):
    kernel = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype=np.float32)

    kernel = kernel / 16.0

    return kernel

def gaussian_smoothing(size, sigma):

    a = size // 2
    x, y = np.mgrid[-a:a+1, -a:a+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)

    return kernel

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

def laplacian(image, mask="sem_diagonais"):
    if mask == 'sem_diagonais':
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        c = -1

    elif mask == 'com_diagonais':
        kernel = np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=np.float32)
        c = -1

    image_float = image.astype(np.float32)

    first_laplacian_image = convolution(image_float, kernel)
    process_laplacian_image = image_float + (c * first_laplacian_image)

    #tom acinzentado
    result_laplacian_image = np.clip(process_laplacian_image, 0, 255).astype(np.uint8)

    return result_laplacian_image

def laplacian_border_detection(image):
    kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
    c = -1

    image_float = image.astype(np.float32)

    first_laplacian_image = convolution(image_float, kernel)
    
    process_laplacian_image = np.abs(first_laplacian_image)

    result_laplacian_image = np.clip(process_laplacian_image, 0, 255).astype(np.uint8)

    return process_laplacian_image

def high_boost(image, k=2.0, size=3):
    image_float = np.asarray(image, dtype=np.float32)

    kernel = smoothing_average(size)
    blurred = convolution(image_float, kernel)

    g_mask = image_float - blurred

    result = image_float + (k * g_mask)

    #tom acinzentado 
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)