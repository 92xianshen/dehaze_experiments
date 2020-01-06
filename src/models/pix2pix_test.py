# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
        'height': tf.io.FixedLenFeature([], tf.int64), 
        'width': tf.io.FixedLenFeature([], tf.int64), 
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        x = tf.io.decode_raw(example['x'], tf.uint8)
        y = tf.io.decode_raw(example['y'], tf.uint8)
        height, width, num_channels = example['height'], example['width'], example['num_channels']

        x = tf.reshape(x, [height, width, num_channels])
        y = tf.reshape(y, [height, width, num_channels])

        x = tf.cast(x, tf.float32)
        x = x / 127.5 - 1
        y = tf.cast(y, tf.float32)
        y = y / 127.5 - 1

        example['x'] = x
        example['y'] = y

        return example
    
    dataset = dataset.map(_parse_function).batch(batch_size)

    return dataset

# def generator_f(dehazy, transmission_map, atmospheric_light):
#     """ f: dehazy |-> hazy, using I(z) = J(z) * t(z) + A(z)(1 - t(z))

#     Args:
#         dehazy: Dehazy image
#         transmission_map: Transmission map
#         atmospheric_light: Atmospheric light
    
#     Returns:
#         Hazy image generated by I(z) = J(z) * t(z) + A(z)(1 - t(z))
#     """
#     hazy = dehazy * transmission_map + atmospheric_light * (1 - transmission_map)
#     return hazy

def haze_func(image, transmission_map, atmospheric_light):
    """ image |-> hazy, using I(z) = J(z) * t(z) + A(z)(1 - t(z))
    Args:
        dehazy: Dehazy image
        transmission_map: Transmission map
        atmospheric_light: Atmospheric light
    
    Returns:
        Hazy image generated by I(z) = J(z) * t(z) + A(z)(1 - t(z))
    """
    hazy = image * transmission_map + atmospheric_light * (1 - transmission_map)
    return hazy

def dehaze_func(hazy, transmission_map, atmospheric_light):
    """ hazy |-> dehazy, using J(z) = (I(z) - A(z)) / t(z) + A(z)
    Args:
        hazy: Hazy image
        transmission_map: Transmission map
        atmospheric_light: Atmospheric light
    
    Returns:
        Dehazy image generated by J(z) = (I(z) - A(z)) / t(z) + A(z)
    """
    transmission_map = tf.abs(transmission_map) + (10**-10)
    dehazy = (hazy - atmospheric_light) / transmission_map + atmospheric_light
    return dehazy

# def dehaze_generator(input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
#     estimator = pix2pix.unet_generator(input_channels, estimation_channels, norm_type)
    
#     inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
#     hazy = inputs
#     estimation = estimator(hazy)
#     transmission_map, atmospheric_light = tf.split(estimation, num_or_size_splits=num_or_size_splits, axis=-1)
#     dehazy = (hazy - atmospheric_light) / transmission_map + atmospheric_light

#     return tf.keras.Model(inputs=inputs, outputs=[dehazy, transmission_map, atmospheric_light])

# def haze_generator(input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
#     estimator = pix2pix.unet_generator(input_channels, estimation_channels, norm_type)

#     inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
#     dehazy = inputs
#     estimation = estimator(dehazy)
#     transmission_map, atmospheric_light = tf.split(estimation, num_or_size_splits=num_or_size_splits, axis=-1)
#     hazy = dehazy * transmission_map + atmospheric_light * (1 - transmission_map)

#     return tf.keras.Model(inputs=inputs, outputs=[hazy, transmission_map, atmospheric_light])

def create_models():
    # g: hazy ---> tran & atp
    generator_g = pix2pix.dehaze_generator(input_channels=3, estimation_channels=6, norm_type='instancenorm', num_or_size_splits=2)
    generator_f = pix2pix.haze_generator(input_channels=3, estimation_channels=6, norm_type='instancenorm', num_or_size_splits=2)
    
    discriminator_x = pix2pix.discriminator(input_channels=3, norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(input_channels=3, norm_type='instancenorm', target=False)

    return generator_g, generator_f, discriminator_x, discriminator_y

def test_one_step():
    dataset = load_dataset(tfrecords_name)
    for record in dataset.take(1):
        sample_hazy, sample_dehazy = record['x'], record['y']
        generator_g, generator_f, discriminator_x, discriminator_y = create_models()

        print('Generator G:', generator_g.summary())

        to_dehazy = generator_g(sample_hazy, training=True)

        to_hazy = generator_f(sample_dehazy, training=True)

        plt.figure(figsize=(8, 8))
        contrast = 8

        imgs = [
            sample_hazy, to_dehazy,
            sample_dehazy, to_hazy,
        ]
        title = [
            'Hazy', 'To dehazy',
            'Dehazy', 'To hazy',  
        ]

        for i in range(len(imgs)):
            plt.subplot(2, 2, i + 1)
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