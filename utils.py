import numpy as np

def getMask(image_width, image_height, left, top, width, height) :

    mask = np.ones((image_height, image_width))

    mask[top : top + height + 1, left : left + width + 1] = 0

    return mask
