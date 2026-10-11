import sys
from PIL import Image

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QImage, QPixmap

import processing.spatial as spatial
from image_io.image_io import read_image, show_image

#show_image(negative(read_image()))
#show_image(thresholding(read_image(), 127))
#show_image(spatial.brightness_adjustment(read_image(), 90))
#show_image(spatial.contrast_adjustment(read_image(), -74))

matriz_imagem = spatial.contrast_adjustment(read_image(), -74)

new_image= Image.fromarray(matriz_imagem)

new_image.save("C:/Users/addso/OneDrive/Área de Trabalho/processadorImagens/tests/bh_contraste.jpg")
print("Imagem salva com sucesso!")

app = QApplication(sys.argv)

height, width, channels = matriz_imagem.shape
bytes_per_line = channels * width

qimage = QImage(
    matriz_imagem.data,
    width,
    height,
    bytes_per_line,
    QImage.Format_RGB888
)

pixmap = QPixmap.fromImage(qimage)

window = QLabel()
window.setPixmap(pixmap)
window.setWindowTitle("Imagem Processada")
window.resize(width, height)
window.show()

sys.exit(app.exec())
