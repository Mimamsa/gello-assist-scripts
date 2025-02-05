"""Image processing module

"""
import numpy as np


def crop_to_4w3h(img):
    """Crop aspect ratio 1.78 to 1.33
    Args
        img (np.ndarray): Image to be cropped, Numpy ndarray of shape (H, W, 3)
    Returns
        (np.ndarray): Cropped image.
    """
    h = img.shape[0]
    w = img.shape[1]
    assert abs(w/h)-1.78 < 0.01, 'Aspect ratio of input image is not 1.78'
    resized_w = int(h/3*4)
    w_offset = int((w-resized_w)/2)
    img = img[:, w_offset:w_offset+resized_w, :]
    return img