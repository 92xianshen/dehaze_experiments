# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf

from skimage import img_as_float32
from skimage.io import imread, imsave

import matplotlib.pyplot as plt

im = np.arange(16).reshape((4, 4))
print(im)
mean_im = cv2.boxFilter(im, cv2.CV_64F, (3, 3), normalize=True)
print(mean_im)

def mean_filter(inputs, radius):
    kernel = tf.constant(1.0 / (radius * radius), dtype=tf.float32, shape=[radius, radius, 1, 1])
    outputs = tf.nn.conv2d(inputs, kernel, strides=1, padding='SAME')

    return outputs

def guided_filter(im, p, r, eps):
    mean_I = mean_filter(im, radius=r)
    mean_p = mean_filter(p, radius=r)
    mean_Ip = mean_filter(im * p, radius=r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = mean_filter(im * im, radius=r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = mean_filter(a, radius=r)
    mean_b = mean_filter(b, radius=r)

    q = mean_a * im + mean_b

    return q


print('GuidedFilter:')

rgb = img_as_float32(imread('test_images/rgb.jpg'))
gt = img_as_float32(imread('test_images/gt.jpg'))

outputs = []
for i in range(rgb.shape[-1]):
    rgb_channel = rgb[..., i]
    gt_channel = gt[..., i]
    outputs += [guided_filter(tf.constant(rgb_channel[np.newaxis, ..., np.newaxis]), tf.constant(gt_channel[np.newaxis, ..., np.newaxis]), 64, 0)]

outputs = tf.concat(outputs, axis=-1)

imsave('test_images/guided_filter_tf.jpg', outputs[0])