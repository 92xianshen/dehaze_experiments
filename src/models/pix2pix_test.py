# -*- coding: utf-8 -*-

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import pix2pix

tfrecords_name = '../../datasets/RICE1.tfrecords'
batch_size = 1

# g: hazy |-> dehazy, using J(z) = (I(z) - A(z)) / t(z) + A(z)
# f: dehazy |-> hazy, using I(z) = J(z) * t(z) + A(z)(1 - t(z))

def load_dataset(tfrecords_name):
    dataset = tf.data.TFRecordDataset([tfrecords_name])

    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string), 
        'y': tf.io.FixedLenFeature([], tf.string), 
        'ale': tf.io.FixedLenFeature([], tf.string), 
        'height': tf.io.FixedLenFeature([], tf.int64), 
        'width': tf.io.FixedLenFeature([], tf.int64), 
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        x = tf.io.decode_raw(example['x'], tf.uint8)
        y = tf.io.decode_raw(example['y'], tf.uint8)
        ale = tf.io.decode_raw(example['ale'], tf.uint8)
        height, width, num_channels = example['height'], example['width'], example['num_channels']

        x = tf.reshape(x, [height, width, num_channels])
        y = tf.reshape(y, [height, width, num_channels])
        ale = tf.reshape(ale, [height, width, num_channels])

        x = tf.cast(x, tf.float32)
        x = x / 127.5 - 1
        y = tf.cast(y, tf.float32)
        y = y / 127.5 - 1
        ale = tf.cast(ale, tf.float32)
        ale = ale / 127.5 - 1

        example['x'] = x
        example['y'] = y
        example['ale'] = ale

        return example
    
    dataset = dataset.map(_parse_function).batch(batch_size)

    return dataset

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

def guided_filter(I, p, r, eps=1e-8):
    """ Guided Filter

    Args:
        I: guidance image
        p: filtering image
        r: the radius of the guidance
        eps: epsilon for the guided filter
    
    Returns:
        Filtering output q
    """
    assert I.shape.ndims == 4 and p.shape.ndims == 4

    I_shape = tf.shape(I)
    p_shape = tf.shape(p)

    # N
    N = box_filter(tf.ones((1, I_shape[1], I_shape[2], 1), dtype=I.dtype), r)

    # mean_x
    mean_I = box_filter(I, r) / N
    # mean_y
    mean_p = box_filter(p, r) / N
    # cov_xy
    cov_Ip = box_filter(I * p, r) / N - mean_I * mean_p
    # var_x
    var_I = box_filter(I * I, r) / N - mean_I * mean_I

    # A
    A = cov_Ip / (var_I + eps)
    # b
    b = mean_p - A * mean_I

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    q = mean_A * I + mean_b

    return q

def create_models():
    # g: hazy ---> tran & atp
    generator_g = pix2pix.dehaze_generator(input_channels=3, estimation_channels=1, norm_type='instancenorm')
    generator_f = pix2pix.haze_generator(input_channels=3, estimation_channels=1, norm_type='instancenorm')
    
    discriminator_x = pix2pix.discriminator(input_channels=3, norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(input_channels=3, norm_type='instancenorm', target=False)

    return generator_g, generator_f, discriminator_x, discriminator_y

def test_one_step():
    dataset = load_dataset(tfrecords_name)
    for record in dataset.take(1):
        sample_hazy, sample_dehazy, sample_ale = record['x'], record['y'], record['ale']

        generator_g, generator_f, discriminator_x, discriminator_y = create_models()

        to_dehazy, tme = generator_g([sample_hazy, sample_ale], training=True)

        to_hazy, hazy_tme = generator_f([sample_dehazy, sample_ale], training=True)

        refined_tme = guided_filter(tf.image.rgb_to_grayscale(sample_hazy), tme, r=60, eps=0.0001)
        refined_to_dehazy = (sample_hazy - sample_ale) / refined_tme + sample_ale

        refined_hazy_tme = guided_filter(tf.image.rgb_to_grayscale(sample_dehazy), hazy_tme, r=60, eps=0.0001)
        refined_to_hazy = sample_dehazy * refined_hazy_tme + 1 * (1 - refined_hazy_tme)
        
        tme = tf.concat([tme] * 3, axis=-1)
        hazy_tme = tf.concat([hazy_tme] * 3, axis=-1)
        refined_tme = tf.concat([refined_tme] * 3, axis=-1)
        refined_hazy_tme = tf.concat([refined_hazy_tme] * 3, axis=-1)

        print(sample_hazy.shape, sample_ale.shape, to_dehazy.shape, tme.shape)
        print(sample_dehazy.shape, to_hazy.shape, hazy_tme.shape)

        plt.figure(figsize=(24, 8))
        contrast = 8

        imgs = [
            sample_hazy, sample_ale, to_dehazy, tme, refined_to_dehazy, refined_tme, 
            sample_dehazy, sample_ale, to_hazy, hazy_tme, refined_to_hazy, refined_hazy_tme,
        ]
        title = [
            'Hazy', 'ALE', 'To dehazy', 'TME', 'Refined to dehazy', 'Refined TME', 
            'Dehazy', 'ALE', 'To hazy', 'Hazy TME', 'Refined to hazy', 'Refined hazy TME'
        ]

        for i in range(len(imgs)):
            plt.subplot(2, 6, i + 1)
            plt.title(title[i])
            plt.imshow(imgs[i][0] * 0.5 + 0.5)
        plt.show()

        plt.figure(figsize=(8, 8))

        plt.subplot(1, 2, 1)
        plt.title('Is a real hazy image?')
        plt.imshow(discriminator_y(sample_hazy)[0, ..., -1], cmap='RdBu_r')

        plt.subplot(1, 2, 2)
        plt.title('Is a real dehazy image?')
        plt.imshow(discriminator_x(sample_dehazy)[0, ..., -1], cmap='RdBu_r')

        plt.show()

if __name__ == "__main__":
    test_one_step()
