# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

from skimage import img_as_float
from skimage.io import imread

import matplotlib.pyplot as plt

from guided_filter import guided_filter

print('GuidedFilter:')
rgb = tf.constant(img_as_float(imread('test_images/rgb.jpg')[np.newaxis, ...]))
gt = tf.constant(img_as_float(imread('test_images/gt.jpg')[np.newaxis, ...]))

output = guided_filter(rgb, gt, 64, 0)

plt.subplot(1, 3, 1)
plt.imshow(rgb.numpy()[0])
plt.title('rgb')

plt.subplot(1, 3, 2)
plt.imshow(gt.numpy()[0])
plt.title('gt')

plt.subplot(1, 3, 3)
plt.imshow(output.numpy()[0])
plt.title('output')

plt.show()