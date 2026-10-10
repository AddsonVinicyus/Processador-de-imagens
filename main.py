from image_io.image_io import read_image, show_image
from processing.spatial import negative, thresholding

#show_image(negative(read_image()))
show_image(thresholding(read_image(), 127))

