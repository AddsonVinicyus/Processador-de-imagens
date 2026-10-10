import numpy as np

def negative(image_array):
    return 255 - image_array

def rgb_to_gray(image_array):
    r = image_array[:,:,0].astype(np.float32)
    g = image_array[:,:,1].astype(np.float32)
    b = image_array[:,:,2].astype(np.float32)

    gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray_image.astype(np.uint8)

def thresholding(image_array, threshold_value):

    if len(image_array.shape) == 3:
        image_array = rgb_to_gray(image_array)
  
    result = np.where(image_array > threshold_value, 255, 0)

    return result.astype(np.uint8)

def brightness_adjustment(image_array, adjustment_value):
    result = np.clip(image_array.astype(np.float32) + adjustment_value, 0, 255)
    return result.astype(np.uint8)
