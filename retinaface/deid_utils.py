import cv2
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def mosaic(image, ratio):
    small = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def elastic(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
    if random_state is None:
        random_state = np.random.RandomState(None)

    #print(random_state)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='nearest')  #wrap,reflect, nearest

    return distored_image.reshape(image.shape)

def shuffle(image):
    np.random.shuffle(image)
    shuffled_image = np.reshape(image, image.shape)
    return shuffled_image

def gaussian_blur(image, kernel=(21,21)):
    blurred_image = cv2.GaussianBlur(image, kernel, 0)
    return blurred_image