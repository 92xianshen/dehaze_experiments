# -*- coding: utf-8 -*-

""" Estimate atmospheric light
"""

import numpy as np
import cv2
import math

def dark_channel(im, sz):
    """ Get the dark channel prior in the (RGB) image data.

    Args:
        im: an numpy array containing data.
        sz: Window size.
    
    Returns:
        The dark channel
    """
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def atmospheric_light(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    A = np.zeros(im.shape)

    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]
    atmsum = atmsum / numpx

    for ind in range(3):
        A[:, :, ind] = atmsum[0, ind]

    return A