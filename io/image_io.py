import numpy as np
from PIL import Image 


def read_image():
    image = Image.open("./testes/bh.jpg")

    if image.mode != 'RGB':
        image = image.convert('RGB')

    matriz_imagem = np.array(image, dtype=np.uint8)

    return matriz_imagem

def show_image(image_array):
    image = Image.fromarray(image_array)
    image.show()


show_image(read_image())

