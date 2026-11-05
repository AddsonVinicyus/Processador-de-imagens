import numpy as np
from processing import (colors, colors_fast, filtering, frequency_tools, histogram,
                        rotation, scale, spatial)

DISPLAY_CLIP = "Cortar (0–255)"
DISPLAY_NORMALIZE = "Normalizar (tom acinzentado)"
DISPLAY_CHOICES = (DISPLAY_CLIP, DISPLAY_NORMALIZE)

LAPLACIAN_KERNELS = {
    "sem_diagonais": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "com_diagonais": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32),
}
LAPLACIAN_MASKS = tuple(LAPLACIAN_KERNELS)

CHANNEL_TYPES = ("R", "G", "B", "I (HSI)")
HIST_ALL = "R, G, B e I (todos)"
HIST_TYPES = ("Tons de cinza",) + CHANNEL_TYPES + (HIST_ALL,)

def to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

def to_gray(img):
    return spatial.rgb_to_gray(img) if img.ndim == 3 else img

def as_rgb(img):
    return img if img.ndim == 3 else np.stack([img] * 3, axis=-1)

def equalize(img):
    equalized, _ = histogram.histogram_equalization(as_rgb(img))
    return equalized

def equalize_hsi(img):
    hsi = colors.rgb2hsi_image(as_rgb(img))
    equalized_hsi, _ = histogram.histogram_equalization_hsi(hsi)
    return colors.hsi2rgb_image(equalized_hsi)

def show_hsi_channel(img, canal):
    hsi = colors.rgb2hsi_image(as_rgb(img))
    channel = colors.hsi_channel_exibition(hsi, canal)
    if canal == "H":
        channel = channel / 360.0  # H está em graus [0, 360)
    return to_u8(channel * 255)

def shift_hue(img, delta):
    hsi = colors.rgb2hsi_image(as_rgb(img))
    return colors.hsi2rgb_image(colors.change_hue(hsi, delta))

def rotate_pivot(img, angulo, usar_centro, px, py):
    pivot = None if usar_centro else (px, py)
    return rotation.rotation_nearest_neighbor_pivot(img, angulo, pivot)

def normalized_histogram(img, tipo):
    """Histograma normalizado (256 valores) de um canal específico."""
    rgb = as_rgb(img)
    if tipo == "I (HSI)":
        hsi = colors_fast.rgb2hsi_array(rgb)
        hist, total = histogram.calc_histogram_hsi(hsi)
    elif tipo in ("R", "G", "B"):
        hist, total = histogram.calc_histogram(rgb[:, :, "RGB".index(tipo)])
    else:
        hist, total = histogram.calc_histogram(img)
    return histogram.calc_histogram_normalized(hist, total)

#CONVOLUÇÃO

def laplacian_filtered(ch, mask):
    """∇²f bruto: tem valores negativos e positivos; nada é cortado."""
    return filtering.convolution(ch.astype(np.float32), LAPLACIAN_KERNELS[mask])

def sobel_component(ch, kernel):
    """Gradiente de Sobel (Gx ou Gy) bruto, com sinal."""
    return filtering.convolution(ch.astype(np.float32), kernel)

def laplacian_sharpen(ch, mask, exibicao):
    """Realce f + c·∇²f (c = -1)."""
    if exibicao == DISPLAY_NORMALIZE:
        return ch.astype(np.float32) - laplacian_filtered(ch, mask)
    return filtering.laplacian(ch, mask)

def high_boost(ch, k, size, exibicao):
    """High-boost f + k·(f − borrada)."""
    if exibicao == DISPLAY_NORMALIZE:
        f = ch.astype(np.float32)
        blurred = filtering.convolution(f, filtering.smoothing_average(size))
        return f + k * (f - blurred)
    return filtering.high_boost(ch, k, size)

