import numpy as np

def proxima_potencia_de_dois(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()

def aplicar_zero_padding(image):
    height, width = image.shape
    new_height = proxima_potencia_de_dois(height)
    new_width = proxima_potencia_de_dois(width)

    pad_height = new_height - height
    pad_width = new_width - width

    image_padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    return image_padded, height, width

def filter_gaussian_lowpass_frequency(height, width, D0):
    center_u = height // 2
    center_v = width // 2

    U, V = np.meshgrid(np.arange(width), np.arange(height))
    D = (U - center_v) ** 2 + (V - center_u) ** 2
    H = np.exp(-D / (2 * D0 ** 2))
    
    return H

def fft_1d(x):
    N = len(x)
    if N <= 1:
        return x.astype(complex)

    even = fft_1d(x[0::2])
    odd = fft_1d(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)

    sup_half = even + factor * odd
    inf_half = even - factor * odd

    return np.concatenate([sup_half, inf_half])

def fft_2d(image):
    matrix = np.asarray(image, dtype=complex)
    height, width = matrix.shape

    rows_result = np.zeros_like(matrix, dtype=complex)

    for i in range(height):
        rows_result[i, :] = fft_1d(matrix[i, :])

    result_2d = np.zeros_like(rows_result, dtype=complex)
    for j in range(width):
        result_2d[:, j] = fft_1d(rows_result[:, j])

    return result_2d

def ifft_2d(complex_spectrum):
    height, width = complex_spectrum.shape
    MN = height * width

    spectrum_conj = np.conj(complex_spectrum)
    spectrum_conj = fft_2d(spectrum_conj)
    image_reconstructed = np.conj(spectrum_conj) / MN

    return np.real(image_reconstructed)

def spectrum_centering(image):
    height, width = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_factor = (-1) ** (x + y)
    return image * center_factor