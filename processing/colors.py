import math
import numpy as np

def rgb2hsi(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    I = (r + g + b) / 3.0

    min_rgb = min(r, g, b)
    if I == 0:
        S = 0
    else:
        S = 1 - (min_rgb / I)

    if S == 0:
        H = 0
    else:
        num = 0.5 * ((r - g) + (r - b))
        den = ((r - g) ** 2 + (r - b) * (g - b)) ** 0.5
        theta = math.acos(num / den) if den != 0 else 0

        if b <= g:
            H = theta * (180 / math.pi)
        else:
            H = (360 - theta * (180 / math.pi))

    return H, S, I

def rgb2hsi_image(image):
    hsi_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            hsi_image[i, j] = rgb2hsi(r, g, b)
    return hsi_image


def hsi2rgb(h, s, i):
    if s == 0:
        r = g = b = i
    else:
        # Garante que h esteja no intervalo [0, 360)
        h = h % 360.0
        
        if 0 <= h < 120:
            # Setor RG (Equações 6.2-5 a 6.2-7)
            b = i * (1 - s)
            r = i * (1 + (s * math.cos(math.radians(h))) / math.cos(math.radians(60 - h)))
            g = 3 * i - (r + b)
            
        elif 120 <= h < 240:
            # Setor GB (Equações 6.2-8 a 6.2-11)
            h_sector = h - 120
            r = i * (1 - s)
            g = i * (1 + (s * math.cos(math.radians(h_sector))) / math.cos(math.radians(60 - h_sector)))
            b = 3 * i - (r + g)
            
        else:
            # Setor BR (Equações 6.2-12 a 6.2-15)
            h_sector = h - 240
            g = i * (1 - s)
            b = i * (1 + (s * math.cos(math.radians(h_sector))) / math.cos(math.radians(60 - h_sector)))
            r = 3 * i - (g + b)
            
    # Converte os valores [0, 1] de volta para [0, 255] e garante os limites
    r = max(0, min(int(r * 255), 255))
    g = max(0, min(int(g * 255), 255))
    b = max(0, min(int(b * 255), 255))

    return r, g, b

def hsi2rgb_image(hsi_image):
    rgb_image = np.zeros_like(hsi_image, dtype=np.uint8)
    for i in range(hsi_image.shape[0]):
        for j in range(hsi_image.shape[1]):
            h, s, i_val = hsi_image[i, j]
            rgb_image[i, j] = hsi2rgb(h, s, i_val)
    return rgb_image

def hsi_channel_exibition(hsi_image, channel):
    channel_image = np.zeros((hsi_image.shape[0], hsi_image.shape[1]), dtype=np.float32)
    
    if channel == 'H':
        channel_image = hsi_image[:, :, 0]
    elif channel == 'S':
        channel_image = hsi_image[:, :, 1]
    elif channel == 'I':
        channel_image = hsi_image[:, :, 2]
    else:
        raise ValueError("Channel must be 'H', 'S', or 'I'")
    
    return channel_image

def change_hue(hsi_image, delta_hue):
    modified_hsi = np.copy(hsi_image)
    modified_hsi[:, :, 0] = (modified_hsi[:, :, 0] + delta_hue) % 360
    return modified_hsi

