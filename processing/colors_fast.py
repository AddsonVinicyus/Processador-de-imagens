import numpy as np

SHADOWS, MIDTONES, HIGHLIGHTS = "Sombras", "Meios-tons", "Realces"
RANGES = (SHADOWS, MIDTONES, HIGHLIGHTS)

# Quanto (de 0 a 1) um valor de ±100 nos controles de balanço desloca o canal
BALANCE_GAIN = 0.5

# Matriz clássica de sépia (escala de cinza amarelada): cada linha gera R, G, B
SEPIA_MATRIX = np.array([[0.393, 0.769, 0.189],
                         [0.349, 0.686, 0.168],
                         [0.272, 0.534, 0.131]], dtype=np.float64)


def _sepia_blend(rgb_float, amount):
    a = float(np.clip(amount, 0.0, 100.0)) / 100.0
    if a == 0.0:
        return rgb_float
    sepia_img = np.clip(rgb_float @ SEPIA_MATRIX.T, 0.0, 1.0)
    return (1.0 - a) * rgb_float + a * sepia_img


def apply_sepia(rgb, amount=100.0):
    out = _sepia_blend(np.asarray(rgb, dtype=np.float64) / 255.0, amount)
    return np.rint(out * 255.0).astype(np.uint8)


def rgb2hsi_array(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    i = (r + g + b) / 3.0
    safe_i = np.where(i > 0, i, 1.0)
    s = np.where(i > 0, 1.0 - rgb.min(axis=-1) / safe_i, 0.0)

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt(np.maximum((r - g) ** 2 + (r - b) * (g - b), 0.0))
    cos_theta = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    h = np.where(b <= g, theta, 360.0 - theta)
    h = np.where(s > 0, h, 0.0)

    return np.stack([h, s, i], axis=-1).astype(np.float32)


def hsi2rgb_array(hsi, as_float=False):
    hsi = np.asarray(hsi, dtype=np.float64)
    h = np.mod(hsi[..., 0], 360.0)
    s = np.clip(hsi[..., 1], 0.0, 1.0)
    i = np.clip(hsi[..., 2], 0.0, 1.0)

    sector = np.minimum((h // 120).astype(int), 2)  # 0: RG, 1: GB, 2: BR
    hh = h - 120.0 * sector

    x = i * (1.0 - s)
    y = i * (1.0 + s * np.cos(np.radians(hh)) / np.cos(np.radians(60.0 - hh)))
    z = 3.0 * i - (x + y)

    r = np.select([sector == 0, sector == 1], [y, x], default=z)
    g = np.select([sector == 0, sector == 1], [z, y], default=x)
    b = np.select([sector == 0, sector == 1], [x, z], default=y)

    rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    return rgb if as_float else np.rint(rgb * 255.0).astype(np.uint8)


def adjust(rgb, hsi=None, hue=0.0, saturation=0.0, brightness=0.0, balance=None, sepia=0.0):
    if hsi is None:
        hsi = rgb2hsi_array(rgb)

    h = (hsi[..., 0] + hue) % 360.0
    s = np.clip(hsi[..., 1] * (1.0 + saturation / 100.0), 0.0, 1.0)
    i = np.clip(hsi[..., 2] + brightness / 100.0, 0.0, 1.0)
    out = hsi2rgb_array(np.stack([h, s, i], axis=-1), as_float=True)

    if balance and any(any(values) for values in balance.values()):
        lum = out.mean(axis=-1)
        weights = {
            SHADOWS: np.clip(1.0 - 2.0 * lum, 0.0, 1.0),
            MIDTONES: 1.0 - np.abs(2.0 * lum - 1.0),
            HIGHLIGHTS: np.clip(2.0 * lum - 1.0, 0.0, 1.0),
        }
        shift = np.zeros_like(out)
        for faixa, values in balance.items():
            # positivo = mais Vermelho / Verde / Azul; negativo = Ciano / Magenta / Amarelo
            delta = np.asarray(values, dtype=np.float64) * (BALANCE_GAIN / 100.0)
            shift += weights[faixa][..., None] * delta
        out = np.clip(out + shift, 0.0, 1.0)

    out = _sepia_blend(out, sepia)

    return np.rint(out * 255.0).astype(np.uint8)