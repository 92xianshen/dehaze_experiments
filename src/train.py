# -*- coding: utf-8 -*-

"""
    Train models via Cycle GAN
    Models:
        Generator G: hazy |-> dehazy, using J(z) = (I(z) - A(z)) / t(z) + A(z)
        Generator F: dehazy |-> hazy, using I(z) = J(z) * t(z) + A(z)(1 - t(z))
        Discriminator X: 
        Discriminator Y: 
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow import keras

from models import pix2pix

tfrecords_name = '../datasets/RICE1.tfrecords'
checkpoint_path = 'checkpoints/train/'
vis_path = 'visualizations/'
batch_size = 1
enable_restoration = 1
num_epochs = 30

buffer_size = batch_size * 4
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
EPSILON = 1e-5

# def dehaze_generator(input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
#     """ Create a generator for dehazing.

#     Args:
#         input_channels: Number of input channels
#         estimation_channels: Number of output channels of UNet generator
#         norm_type: Normalization type, either 'batchnorm' or 'instancenorm'
#         num_or_size_splits: Split the estimation of UNet generator

#     Returns:
#         Callable Keras model for dehazing
#     """
#     estimator = pix2pix.unet_generator(input_channels, estimation_channels, norm_type)
    
#     inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
#     hazy = inputs
#     estimation = estimator(hazy)
#     transmission_map, atmospheric_light = tf.split(estimation, num_or_size_splits=num_or_size_splits, axis=-1)
#     dehazy = (hazy - atmospheric_light) / transmission_map + atmospheric_light

#     return tf.keras.Model(inputs=inputs, outputs=dehazy)

# class DehazeGenerator(tf.keras.Model):
#     """ Create a generator for dehazing. """
#     def __init__(self, input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
#         super(DehazeGenerator, self).__init__()
#         self.estimator = pix2pix.unet_generator(input_channels, estimation_channels, norm_type)
#         self.num_or_size_splits = num_or_size_splits
    
#     def call(self, hazy, training=True):
#         estimation = self.estimator(hazy, training=training)
#         transmission_map, atmospheric_light = tf.split(estimation, num_or_size_splits=self.num_or_size_splits, axis=-1)
#         transmission_map += EPSILON # Avoid to be devided by zero
#         dehazy = (hazy - atmospheric_light) / transmission_map + atmospheric_light
        
#         return dehazy

# def haze_generator(input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
#     """ Create a generator for hazing.

#     Args:
#         input_channels: Number of input channels
#         estimation_channels: Number of output channels of UNet generator
#         norm_type: Normalization type, either 'batchnorm' or 'instancenorm'
#         num_or_size_splits: Split the estimation of UNet generator

#     Returns:
#         Callable Keras model for hazing
#     """
#     estimator = pix2pix.unet_generator(input_channels, estimation_channels, norm_type)

#     inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
#     dehazy = inputs
#     estimation = estimator(dehazy)
#     transmission_map, atmospheric_light = tf.split(estimation, num_or_size_splits=num_or_size_splits, axis=-1)
#     hazy = dehazy * transmission_map + atmospheric_light * (1 - transmission_map)

#     return tf.keras.Model(inputs=inputs, outputs=hazy)

# class HazeGenerator(tf.keras.Model):
#     """ Create a generator for dehazing. """
#     def __init__(self, input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
#         super(HazeGenerator, self).__init__()
#         self.estimator = pix2pix.unet_generator(input_channels, estimation_channels, norm_type)
#         self.num_or_size_splits = num_or_size_splits
    
#     def call(self, dehazy, training=True):
#         estimation = self.estimator(dehazy, training=training)
#         transmission_map, atmospheric_light = tf.split(estimation, num_or_size_splits=self.num_or_size_splits, axis=-1)
#         hazy = dehazy * transmission_map + atmospheric_light * (1 - transmission_map)

#         return hazy

def load_dataset(tfrecords_name):
    """ Load a dataset and parse the records within the dataset.
    
    Args:
        tfrecords_name: File name of tfrecords
    
    Returns:
        A TFRecordDataset
    """
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

def generator_f(dehazy, transmission_map, atmospheric_light):
    """ Define the generator f: dehazy |-> hazy, using I(z) = J(z) * t(z) + A(z)(1 - t(z))
    Note that there is no trainable variable inside

    Args:
        dehazy: Dehazy image
        transmission_map: Transmission map
        atmospheric_light: Atmospheric light
    
    Returns:
        Hazy image generated by I(z) = J(z) * t(z) + A(z)(1 - t(z))
    """
    hazy = dehazy * transmission_map + atmospheric_light * (1 - transmission_map)
    return hazy

def haze_func(image, transmission_map, atmospheric_light):
    """ Define a transformation from clear to hazy image: image |-> hazy, using I(z) = J(z) * t(z) + A(z)(1 - t(z))
    
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
    """ Define transformation from hazy to dehazy images: hazy |-> dehazy, using J(z) = (I(z) - A(z)) / t(z) + A(z)
    
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

def discriminator_loss(real, generated):
    """ Define discriminator loss

    Args: 
        real: Real input
        generated: Generated input
    
    Returns:
        Discriminator loss
    """
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss

    return 0.5 * total_disc_loss

def generator_loss(generated):
    """ Define generator loss

    Args: 
        generated: Generated input

    Returns:
        Generator loss
    """
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real, cycled):
    """ Define cycle-consistency loss between real and cycled image

    Args:
        real: Real input
        cycled: Cycled input
    
    Returns:
        Cycle-consistency loss
    """
    l1_loss = tf.reduce_mean(tf.abs(real - cycled))

    return LAMBDA * l1_loss

def identity_loss(real, same):
    """ Define identity loss between real image and image that should be the same as the real one. 

    Args:
        real: Real input
        same: Input which should be the same as the real input
    
    Returns:
        Identity loss
    """
    loss = tf.reduce_mean(tf.abs(real - same))
    return LAMBDA * 0.5 * loss

def train_loop():
    # Load dataset
    dataset = load_dataset(tfrecords_name)

    # For visualization
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    # Generator G: hazy |-> dehazy
    generator_g = pix2pix.dehaze_generator(input_channels=3, estimation_channels=6, norm_type='instancenorm', num_or_size_splits=2)
    # Generator F: dehazy |-> hazy
    generator_f = pix2pix.haze_generator(input_channels=3, estimation_channels=6, norm_type='instancenorm', num_or_size_splits=2)
    # Discriminator for hazy: discriminate real or fake hazy
    discriminator_hazy = pix2pix.discriminator(input_channels=3, norm_type='instancenorm', target=False)
    # Discriminator for dehazy: discriminate real or fake dehazy
    discriminator_dehazy = pix2pix.discriminator(input_channels=3, norm_type='instancenorm', target=False)

    # Optimizers
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_hazy_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_dehazy_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Checkpoints and manager
    ckpt = tf.train.Checkpoint(
        generator_g=generator_g, 
        generator_f=generator_f,
        discriminator_hazy=discriminator_hazy,
        discriminator_dehazy=discriminator_dehazy,
        generator_g_optimizer=generator_g_optimizer,
        generator_f_optimizer=generator_f_optimizer,
        discriminator_hazy_optimizer=discriminator_hazy_optimizer,
        discriminator_dehazy_optimizer=discriminator_dehazy_optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if restoration is enabled and a checkpoint exists, restore the latest checkpoint.
    if enable_restoration and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[{}] Latest checkpoint restored.'.format(time.asctime()))

    @tf.function
    def train_step(real_hazy, real_dehazy):
        """ Define one train step for the training loop.

        Args:
            hazy: Hazy image
            dehazy: Dehazy image
        
        Returns:
            None
        """
        # persistent is set to True because the tape is used more than 
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates hazy -> dehazy.
            # Generator F translates dehazy -> hazy.
            fake_dehazy = generator_g(real_hazy, training=True)
            cycled_hazy = generator_f(fake_dehazy, training=True)

            fake_hazy = generator_f(real_dehazy, training=True)
            cycled_dehazy = generator_g(fake_dehazy, training=True)

            # same_hazy and same_dehazy are used for identity loss.
            same_hazy = generator_f(real_hazy, training=True)
            same_dehazy = generator_g(real_dehazy, training=True)

            disc_real_hazy = discriminator_hazy(real_hazy, training=True)
            disc_real_dehazy = discriminator_dehazy(real_dehazy, training=True)

            disc_fake_hazy = discriminator_hazy(fake_hazy, training=True)
            disc_fake_dehazy = discriminator_dehazy(fake_dehazy, training=True)

            # Calculate the loss.
            gen_g_loss = generator_loss(disc_fake_dehazy)
            gen_f_loss = generator_loss(disc_fake_hazy)

            total_cycle_loss = calc_cycle_loss(real_hazy, cycled_hazy) + calc_cycle_loss(real_dehazy, cycled_dehazy)

            # Total generator loss = adversarial loss + cycle loss.
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_dehazy, same_dehazy)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_hazy, same_hazy)

            disc_hazy_loss = discriminator_loss(disc_real_hazy, disc_fake_hazy)
            disc_dehazy_loss = discriminator_loss(disc_real_dehazy, disc_fake_dehazy)

        # Calculate the gradients for generators and discriminators.
        generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
        discriminator_hazy_gradients = tape.gradient(disc_hazy_loss, discriminator_hazy.trainable_variables)
        discriminator_dehazy_gradients = tape.gradient(disc_dehazy_loss, discriminator_dehazy.trainable_variables)

        # Apply the gradients to the optimizers
        generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
        generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
        discriminator_hazy_optimizer.apply_gradients(zip(discriminator_hazy_gradients, discriminator_hazy.trainable_variables))
        discriminator_dehazy_optimizer.apply_gradients(zip(discriminator_dehazy_gradients, discriminator_dehazy.trainable_variables))

    for epoch in range(num_epochs):
        start = time.time()
        n = 0
        dataset = dataset.shuffle(buffer_size=buffer_size)
        for record in dataset.take(-1):
            hazy, dehazy = record['x'], record['y']
            train_step(hazy, dehazy)
            
            if n % 10 == 0:
                print('.', end='')
            n += 1
        print('')
        print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        ckpt_save_path = ckpt_manager.save()
        print('Checkpoint for epoch {} saved at {}'.format(epoch + 1, ckpt_save_path))

        # Train visualization
        for record in dataset.take(1):
            hazy, dehazy = record['x'], record['y']
            to_dehazy = generator_g(hazy, training=False)
            to_hazy = generator_f(dehazy, training=False)
            imgs = [
                hazy, to_dehazy,
                dehazy, to_hazy,
            ]
            titles = [
                'Hazy', 'To dehazy', 
                'Dehazy', 'To hazy',
            ]

            plt.figure(figsize=(8, 8))
            for i in range(len(imgs)):
                plt.subplot(2, 2, i + 1)
                plt.title(titles[i])
                plt.imshow(imgs[i][0] * 0.5 + 0.5)
            plt.savefig(os.path.join(vis_path, 'epoch_{}.png'.format(epoch)))


if __name__ == "__main__":
    train_loop()