# -*- coding: utf-8 -*-

import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
    """ Instance Normalization Layer (https://arxiv.org/abs/1607.08022). """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', 
            shape=input_shape[-1:], 
            initializer=tf.random_normal_initializer(1., 0.02), 
            trainable=True
        )

        self.offset = self.add_weight(
            name='offset', 
            shape=input_shape[-1:], 
            initializer='zeros', 
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """ Downsamples an input.

    Conv2D => Batchnorm => LeakyReLU

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer

    Returns:
        Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same', 
            kernel_initializer=initializer, use_bias=False
        )
    )
    
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """ Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => ReLU

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'
        apply_dropout: If True, adds the dropout layer

    Returns:
        Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, 
            padding='same', kernel_initializer=initializer, use_bias=False
        )
    )

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())

    return result

# def unet_generator(input_channels, output_channels, norm_type='batchnorm'):
#     """ Modified u-net generator model (https://arxiv.org/abs/1611.07004).

#     Args:
#         input_channels: Input channels
#         output_channels: Output channels
#         norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

#     Returns:
#         Generator model
#     """

#     # input size is 512 x 512
#     down_stack = [
#         downsample(64, 4, norm_type, apply_norm=False), # (bs, 256, 256, 64)
#         downsample(128, 4, norm_type), # (bs, 128, 128, 128)
#         downsample(256, 4, norm_type), # (bs, 64, 64, 256)
#         downsample(512, 4, norm_type), # (bs, 32, 32, 512)
#         downsample(512, 4, norm_type), # (bs, 16, 16, 512)
#         downsample(512, 4, norm_type), # (bs, 8, 8, 512)
#         downsample(512, 4, norm_type), # (bs, 4, 4, 512)
#         downsample(512, 4, norm_type), # (bs, 2, 2, 512)
#         downsample(512, 4, norm_type), # (bs, 1, 1, 512)
#     ]

#     up_stack = [
#         upsample(512, 4, norm_type, apply_dropout=True), # (bs, 2, 2, 1024)
#         upsample(512, 4, norm_type, apply_dropout=True), # (bs, 4, 4, 1024)
#         upsample(512, 4, norm_type, apply_dropout=True), # (bs, 8, 8, 1024)
#         upsample(512, 4, norm_type), # (bs, 16, 16, 1024)
#         upsample(512, 4, norm_type), # (bs, 32, 32, 1024)
#         upsample(256, 4, norm_type), # (bs, 64, 64, 512)
#         upsample(128, 4, norm_type), # (bs, 128, 128, 256)
#         upsample(64, 4, norm_type), # (bs, 256, 256, 128)
#     ]

#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = tf.keras.layers.Conv2DTranspose(
#         output_channels, 4, strides=2,
#         padding='same', kernel_initializer=initializer, 
#         activation='tanh'
#     ) # (bs, 512, 512, output_channels)

#     concat = tf.keras.layers.Concatenate()

#     inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
#     x = inputs

#     # Downsample through the model
#     skips = []
#     for down in down_stack:
#         x = down(x)
#         skips.append(x)
    
#     skips = reversed(skips[:-1])

#     # Upsample and establishing the skip connections
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         x = concat([x, skip])

#     x = last(x)

#     return tf.keras.Model(inputs=inputs, outputs=x)

def dehaze_generator(input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
    """ Create a generator for dehazing.

    Args:
        input_channels: Number of input channels
        estimation_channels: Number of output channels of UNet generator
        norm_type: Normalization type, either 'batchnorm' or 'instancenorm'
        num_or_size_splits: Split the estimation of UNet generator

    Returns:
        Callable Keras model for dehazing
    """

    # input size is 512 x 512
    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False), # (bs, 256, 256, 64)
        downsample(128, 4, norm_type), # (bs, 128, 128, 128)
        downsample(256, 4, norm_type), # (bs, 64, 64, 256)
        downsample(512, 4, norm_type), # (bs, 32, 32, 512)
        downsample(512, 4, norm_type), # (bs, 16, 16, 512)
        downsample(512, 4, norm_type), # (bs, 8, 8, 512)
        downsample(512, 4, norm_type), # (bs, 4, 4, 512)
        downsample(512, 4, norm_type), # (bs, 2, 2, 512)
        downsample(512, 4, norm_type), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4, norm_type), # (bs, 16, 16, 1024)
        upsample(512, 4, norm_type), # (bs, 32, 32, 1024)
        upsample(256, 4, norm_type), # (bs, 64, 64, 512)
        upsample(128, 4, norm_type), # (bs, 128, 128, 256)
        upsample(64, 4, norm_type), # (bs, 256, 256, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        estimation_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer, 
        activation='tanh'
    ) # (bs, 512, 512, estimation_channels)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
    x, hazy = inputs, inputs

    # Downsample through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])

    # Upsample and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    # Dehaze using equation 
    transmission_map, atmospheric_light = tf.split(x, num_or_size_splits=num_or_size_splits, axis=-1)
    dehazy = (hazy - atmospheric_light) / (transmission_map + 1e-5) + atmospheric_light
    dehazy = tf.keras.activations.tanh(dehazy)

    return tf.keras.Model(inputs=inputs, outputs=dehazy)

def haze_generator(input_channels=3, estimation_channels=6, norm_type='batchnorm', num_or_size_splits=2):
    """ Create a generator for hazing.

    Args:
        input_channels: Number of input channels
        estimation_channels: Number of output channels of UNet generator
        norm_type: Normalization type, either 'batchnorm' or 'instancenorm'
        num_or_size_splits: Split the estimation of UNet generator

    Returns:
        Callable Keras model for hazing
    """

    # input size is 512 x 512
    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False), # (bs, 256, 256, 64)
        downsample(128, 4, norm_type), # (bs, 128, 128, 128)
        downsample(256, 4, norm_type), # (bs, 64, 64, 256)
        downsample(512, 4, norm_type), # (bs, 32, 32, 512)
        downsample(512, 4, norm_type), # (bs, 16, 16, 512)
        downsample(512, 4, norm_type), # (bs, 8, 8, 512)
        downsample(512, 4, norm_type), # (bs, 4, 4, 512)
        downsample(512, 4, norm_type), # (bs, 2, 2, 512)
        downsample(512, 4, norm_type), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4, norm_type), # (bs, 16, 16, 1024)
        upsample(512, 4, norm_type), # (bs, 32, 32, 1024)
        upsample(256, 4, norm_type), # (bs, 64, 64, 512)
        upsample(128, 4, norm_type), # (bs, 128, 128, 256)
        upsample(64, 4, norm_type), # (bs, 256, 256, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        estimation_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer, 
        activation='tanh'
    ) # (bs, 512, 512, output_channels)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
    x, dehazy = inputs, inputs

    # Downsample through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])

    # Upsample and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    transmission_map, atmospheric_light = tf.split(x, num_or_size_splits=num_or_size_splits, axis=-1)
    hazy = dehazy * transmission_map + atmospheric_light * (1 - transmission_map)
    hazy = tf.keras.activations.tanh(hazy)

    return tf.keras.Model(inputs=inputs, outputs=hazy)

def discriminator(input_channels, norm_type='batchnorm', target=True):
    """ PatchGAN discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
        input_channels: Input channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        target: Bool, indicating whether target image is an input or not.

    Returns:
        Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, input_channels], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, input_channels], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 512, 512, channels * 2)

    down1 = downsample(64, 4, norm_type, False)(x) # (bs, 256, 256, 64)
    down2 = downsample(128, 4, norm_type)(down1) # (bs, 128, 128, 128)
    down3 = downsample(256, 4, norm_type)(down2) # (bs, 64, 64, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 66, 66, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, 
        use_bias=False
    )(zero_pad1) # (bs, 63, 63, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 64, 64, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, 
        kernel_initializer=initializer
    )(zero_pad2) # (bs, 61, 61, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)
