# -*- coding: utf-8 -*-

""" Estimate atmospheric light
"""

import numpy as np
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

# def dark_channel(im, sz):
#     """ Get the dark channel prior in the (RGB) image data.

#     Args:
#         im: an numpy array containing data.
#         sz: Window size.
    
#     Returns:
#         The dark channel
#     """
#     b, g, r = cv2.split(im)
#     dc = cv2.min(cv2.min(r, g), b)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
#     dark = cv2.erode(dc, kernel)
#     return dark

# def atmospheric_light(im, dark):
#     [h, w] = im.shape[:2]
#     imsz = h * w
#     numpx = int(max(math.floor(imsz / 1000), 1))
#     darkvec = dark.reshape(imsz, 1)
#     imvec = im.reshape(imsz, 3)

#     indices = darkvec.argsort()
#     indices = indices[imsz - numpx::]

#     atmsum = np.zeros([1, 3])
#     A = np.zeros(im.shape)

#     for ind in range(1, numpx):
#         atmsum = atmsum + imvec[indices[ind]]
#     atmsum = atmsum / numpx

#     for ind in range(3):
#         A[:, :, ind] = atmsum[0, ind]

#     return A


def calc_dark_channel(ims, w):
    """ Get the dark channel prior in the (RGB) image data.

    Args:
        ims: a 4-D tensor containing data.
        w: Window size.
    
    Returns:
        The dark channel
    """
    batch_size, height, width, num_channels = tf.shape(ims)
    paddeds = tf.pad(ims, [[0, 0], [w // 2, w // 2], [w // 2, w // 2], [0, 0]], 'reflect')
    darkchs = tf.variable((batch_size, height, width, 1))
    for i in range(batch_size):
        for j in range(height):
            for k in range(width):
                darkchs[i, j, k, 0] = tf.reduce_min(paddeds[i, j:j + w, k:k + w])
    return darkchs

def calc_atmospheric_light(ims, darks):
    """ Estimate the global atmospheric light from a hazy image, using the 0.1% brightest pixels of the dark channel
    """
    batch_size, height, width, num_channels = tf.shape(ims)
    imsz = height * width
    numpx = int(tf.reduce_max(tf.floor(imsz / 1000), 1))
    darkvecs = tf.reshape(darks, [batch_size, -1, 1])
    imvecs = tf.reshape(ims, [batch_size, -1, 3])

    indices = [tf.argsort(darkvec) for darkvec in darkvecs]
    indices = indices[:, imsz - numpx::]

    atmsum = tf.zeros([batch_size, 1, 3])
    A = tf.zeros(ims.shape)

    for i in range(batch_size):
        for ind in range(numpx):
            atmsum[i] = atmsum[i] + imvecs[i, indices[ind]]
        atmsum[i] = atmsum[i] / numpx

    for i in range(batch_size):
        for ind in range(num_channels):
            A[i, ..., ind] = atmsum[i, 0, ind]

    return A


if __name__ == "__main__":
    im = np.asarray(Image.open('test_images/15.png')) / 255.0
    ims = tf.constant(im[np.newaxis])
    print(ims.shape)

    darks = calc_dark_channel(ims, w=15)
    As = calc_atmospheric_light(ims, darks)

    plt.subplot(1, 3, 1)
    plt.imshow(ims[0])
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(darks[0])
    plt.title('Dark channel')

    plt.subplot(1, 3, 3)
    plt.imshow(As[0])
    plt.title('Atmospheric light')

    plt.show()
