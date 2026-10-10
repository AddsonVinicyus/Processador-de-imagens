import processing.spatial as spatial
from image_io.image_io import read_image, show_image

#show_image(negative(read_image()))
#show_image(thresholding(read_image(), 127))
show_image(spatial.brightness_adjustment(read_image(), 90))