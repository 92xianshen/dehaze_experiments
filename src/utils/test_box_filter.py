# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf

from skimage import img_as_float
from skimage.io import imread

from guided_filter import box_filter, guided_filter

print('BoxFilter:')

inputs = tf.constant(np.reshape(np.arange(1, 73), (1, 8, 9, 1)))
outputs = box_filter(inputs, 3)

assert np.isclose(outputs.numpy().mean(), 1137.6, 0.1)
assert np.isclose(outputs.numpy().std(), 475.2, 0.1)

im = tf.constant(img_as_float(imread('test_images/rgb.jpg')[np.newaxis, ...]))
r = box_filter(im, 64)

assert np.isclose(r[..., 0].numpy().mean(), 10305.0, 0.1)
assert np.isclose(r[..., 0].numpy().std(),  2206.4,  0.1)
assert np.isclose(r[..., 1].numpy().mean(), 7536.0,  0.1)
assert np.isclose(r[..., 1].numpy().std(),  2117.0,  0.1)
assert np.isclose(r[..., 2].numpy().mean(), 6203.0,  0.1)
assert np.isclose(r[..., 2].numpy().std(),  2772.3,  0.1)