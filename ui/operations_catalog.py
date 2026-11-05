import numpy as np
from ui.models import Param, Op
from processing import (colors, colors_fast, filtering, frequency_tools, histogram,
                        rotation, scale, spatial)
from ui.util.image_utils import (
    DISPLAY_CHOICES,
    DISPLAY_CLIP,
    HIST_TYPES,
    LAPLACIAN_MASKS,
    as_rgb,
    equalize,
    equalize_hsi,
    high_boost,
    laplacian_filtered,
    laplacian_sharpen,
    rotate_pivot,
    shift_hue,
    show_hsi_channel,
    sobel_component,
    to_gray,
    to_u8,
)

def size_param(default=3):
    return Param("size", "Kernel", "int", default, 3, 31, 2, odd=True)

SPATIAL_OPS = {
    "Negativo": Op(lambda img: spatial.negative(img)),
    "Limiarização": Op(
        lambda img, limiar: spatial.thresholding(img, limiar),
        params=[Param("limiar", "Limiar", "int", 128, 0, 255)],
    ),
    "Brilho": Op(
        lambda img, valor: spatial.brightness_adjustment(img, valor),
        params=[Param("valor", "Ajuste", "int", 50, -255, 255)],
    ),
    "Contraste": Op(
        lambda img, valor: spatial.contrast_adjustment(img, valor),
        params=[Param("valor", "Ajuste", "int", 40, -127, 254)],
    ),
}

FILTER_OPS = {
    "Suavização – média": Op(
        lambda ch, size: to_u8(filtering.convolution(ch, filtering.smoothing_average(size))),
        "filter",
        [size_param()],
    ),
    "Suavização – média ponderada (3x3)": Op(
        lambda ch: to_u8(filtering.convolution(ch, filtering.smoothing_weighted_average(3))),
        "filter",
    ),
    "Suavização – gaussiana": Op(
        lambda ch, size, sigma: to_u8(
            filtering.convolution(ch, filtering.gaussian_smoothing(size, sigma))
        ),
        "filter",
        [size_param(5), Param("sigma", "Sigma", "float", 1.0, 0.1, 20, 0.1)],
    ),
    "Filtro da mediana": Op(
        lambda ch, size: to_u8(filtering.median_filter(ch, size)),
        "filter",
        [size_param()],
    ),
    "Laplaciano (realce)": Op(
        laplacian_sharpen,
        "filter",
        [
            Param("mask", "Máscara", "choice", "sem_diagonais", choices=LAPLACIAN_MASKS),
            Param("exibicao", "Exibição", "choice", DISPLAY_CLIP, choices=DISPLAY_CHOICES),
        ],
    ),
    "High-boost": Op(
        high_boost,
        "filter",
        [
            Param("k", "k", "float", 2.0, 0.1, 10, 0.1),
            size_param(),
            Param("exibicao", "Exibição", "choice", DISPLAY_CLIP, choices=DISPLAY_CHOICES),
        ],
    ),
    "Laplaciano (imagem filtrada)": Op(
        laplacian_filtered,
        "edge",
        [Param("mask", "Máscara", "choice", "sem_diagonais", choices=LAPLACIAN_MASKS)],
    ),
    "Sobel (magnitude)": Op(lambda g: filtering.sobel_magnitude(g), "edge"),
    "Sobel – Gx (horizontal)": Op(
        lambda g: sobel_component(g, filtering.sobel_x()), "edge"
    ),
    "Sobel – Gy (vertical)": Op(
        lambda g: sobel_component(g, filtering.sobel_y()), "edge"
    ),
}

HISTOGRAM_OPS = {
    "Exibir histograma": Op(
        lambda app, tipo: app.show_histogram(tipo),
        "action",
        [Param("tipo", "Canal", "choice", HIST_TYPES[0], choices=HIST_TYPES)],
    ),
    "Equalização (tons de cinza)": Op(equalize),
    "Equalização do canal I (HSI)": Op(equalize_hsi),
}

COLOR_OPS = {
    "Converter para tons de cinza": Op(lambda img: to_gray(img)),
    "Converter para colorido (RGB)": Op(lambda img: as_rgb(img)),
    "Ajuste interativo HSI (tempo real)": Op(
        lambda app: app.open_color_adjust(), "action"
    ),
    "Sépia": Op(
        lambda img, intensidade: colors_fast.apply_sepia(as_rgb(img), intensidade),
        params=[Param("intensidade", "Intensidade (%)", "int", 100, 0, 100, 5)],
    ),
    "Exibir canal H / S / I": Op(
        show_hsi_channel,
        params=[Param("canal", "Canal", "choice", "H", choices=("H", "S", "I"))],
    ),
    "Alterar matiz (Hue)": Op(
        shift_hue,
        params=[Param("delta", "Δ Hue (°)", "int", 60, -360, 360, 10)],
    ),
}

GEOMETRY_OPS = {
    "Rotação – vizinho mais próximo": Op(
        lambda img, angulo: rotation.rotation_nearest_neighbor(img, angulo),
        params=[Param("angulo", "Ângulo (°)", "float", 45.0, -360, 360, 5)],
    ),
    "Rotação – bilinear": Op(
        lambda img, angulo: rotation.rotation_bilinear(img, angulo),
        params=[Param("angulo", "Ângulo (°)", "float", 45.0, -360, 360, 5)],
    ),
    "Rotação – vizinho mais próximo (pivô)": Op(
        rotate_pivot,
        params=[
            Param("angulo", "Ângulo (°)", "float", 45.0, -360, 360, 5),
            Param("usar_centro", "Pivô no centro", "bool", True),
            Param("px", "Pivô x", "int", 0, 0, 10000, 10),
            Param("py", "Pivô y", "int", 0, 0, 10000, 10),
        ],
    ),
    "Escala – vizinho mais próximo": Op(
        lambda img, fator: scale.scale_nearest_neighbor(img, fator),
        params=[Param("fator", "Fator", "float", 2.0, 0.1, 4.0, 0.1)],
    ),
    "Escala – bilinear": Op(
        lambda img, fator: scale.scale_bilinear(img, fator),
        params=[Param("fator", "Fator", "float", 2.0, 0.1, 4.0, 0.1)],
    ),
}

FREQUENCY_OPS = {
    "Exibir espectro de Fourier": Op(lambda app: app.show_spectrum(), "action"),
    "Editor de espectro (filtragem manual)": Op(
        lambda app: app.open_spectrum_editor(), "action"
    ),
    "Passa-baixa gaussiano": Op(
        lambda ch, d0: frequency_tools.gaussian_lowpass(ch, d0),
        "filter",
        [Param("d0", "D0 (raio de corte)", "int", 30, 1, 500, 5)],
    ),
}

# Mapeamento das abas

TABS = {
    "Espaciais": (SPATIAL_OPS, False),
    "Filtros": (FILTER_OPS, True),
    "Histograma": (HISTOGRAM_OPS, False),
    "Cores": (COLOR_OPS, False),
    "Geometria": (GEOMETRY_OPS, False),
    "Frequência": (FREQUENCY_OPS, True),
}

def run_operation(op, img, params, per_channel):
    #Executa a operação apropriada de acordo com o seu tipo (point, filter, edge).
    if op.kind == "point":
        result = op.fn(img, **params)
    elif op.kind == "edge" or not per_channel or img.ndim == 2:
        result = op.fn(to_gray(img), **params)
    else:
        result = np.stack([op.fn(img[:, :, c], **params) for c in range(3)], axis=-1)
    result = np.asarray(result)
    if np.issubdtype(result.dtype, np.floating):
        return result  # matriz bruta (pode ter negativos e > 255): normalizada só na exibição
    return result.astype(np.uint8)

def describe(name, op, params):
    #Gera o texto para o histórico indicando o nome da operação e parâmetros usados.
    if not op.params:
        return name
    parts = []
    for p in op.params:
        value = params[p.key]
        if p.kind == "bool":
            value = "sim" if value else "não"
        elif p.kind == "float":
            value = f"{value:g}"
        parts.append(f"{p.label} {value}")
    return f"{name} ({', '.join(parts)})"