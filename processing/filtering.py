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

    convolved_image = np.clip(convolved_image, 0, 255)

    return convolved_image.astype(np.uint8)