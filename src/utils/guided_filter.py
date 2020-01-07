# -*- coding: utf-8 -*-

""" Implementation for Guided Image Filtering

Reference:
    http://research.microsoft.com/en-us/um/people/kahe/eccv10/
    https://github.com/wuhuikai/DeepGuidedFilter
"""

import cv2
import numpy as np
import tensorflow as tf


def diff_x(inputs, r):
    assert inputs.shape.ndims == 4

    left    = inputs[:,         r:2 * r + 1]
    middle  = inputs[:, 2 * r + 1:         ] - inputs[:,           :-2 * r - 1]
    right   = inputs[:,        -1:         ] - inputs[:, -2 * r - 1:    -r - 1]

    outputs = tf.concat([left, middle, right], axis=1)

    return outputs

def diff_y(inputs, r):
    assert inputs.shape.ndims == 4

    left    = inputs[:, :,         r:2 * r + 1]
    middle  = inputs[:, :, 2 * r + 1:         ] - inputs[:, :,           :-2 * r - 1]
    right   = inputs[:, :,        -1:         ] - inputs[:, :, -2 * r - 1:    -r - 1]

    outputs = tf.concat([left, middle, right], axis=2)

    return outputs

def box_filter(x, r):
    assert x.shape.ndims == 4

    return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=1), r), axis=2), r)

def guided_filter(x, y, r, eps=1e-8):
    assert x.shape.ndims == 4 and y.shape.ndims == 4

    x_shape = tf.shape(x)
    y_shape = tf.shape(y)

    # N
    N = box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    # mean_x
    mean_x = box_filter(x, r) / N
    # mean_y
    mean_y = box_filter(y, r) / N
    # cov_xy
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    # var_x
    var_x = box_filter(x * x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    outputs = mean_A * x + mean_b

    return outputs

def fast_guided_filter(lr_x, lr_y, hr_x, r, eps=1e-8):
    assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4

    lr_x_shape = tf.shape(lr_x)
    lr_y_shape = tf.shape(lr_y)
    hr_x_shape = tf.shape(hr_x)

    # N
    N = box_filter(tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    # mean_x
    mean_x = box_filter(lr_x, r) / N
    # mean_y
    mean_y = box_filter(lr_y, r) / N
    # cov_xy
    cov_xy = box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    # var_x
    var_x = box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = tf.image.resize(A, hr_x_shape[1:3])
    mean_b = tf.image.resize(b, hr_x_shape[1:3])

    outputs = mean_A * hr_x + mean_b

    return outputs

if __name__ == "__main__":
    I = tf.random.normal([1, 512, 512, 3])
    p = tf.random.normal([1, 512, 512, 3])

    q_tf = guided_filter(I, p, r=60)
    q_cv = cv2.ximgproc.guidedFilter(I.numpy()[0], p.numpy()[0], 60, eps=1e-8)

    print(np.mean((q_tf - q_cv)**2))
