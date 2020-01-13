# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pix2pix

tfrecords_name = '../../../datasets/RICE1.tfrecords'
checkpoint_path = 'checkpoints/train/'
batch_size = 1
height = 512
width = 512

buffer_size = batch_size * 4

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

# def generate_test_set(record):
#     hazys, dehazys = list(), list()
    
#     hazy, dehazy, ale = record['x'], record['y'], record['ale']
#     hazys += [h for h in hazy]
#     dehazys += [deh for deh in dehazy]

#     hazys = tf.stack(hazys, axis=0)
#     dehazys = tf.stack(dehazys, axis=0)

#     hazy_labels, dehazy_labels = tf.ones(shape=tf.shape(hazys)[0]), tf.zeros(shape=tf.shape(dehazys)[0])

#     images = tf.concat([hazys, dehazys], axis=0)
#     labels = tf.concat([hazy_labels, dehazy_labels], axis=0)

#     indices = tf.range(tf.shape(images)[0])
#     indices = tf.random.shuffle(indices)
#     images = tf.gather(images, indices)
#     labels = tf.gather(labels, indices)

#     return images, labels

def generate_test_set(dataset):
    hazys, dehazys = list(), list()
    
    for record in dataset:
        hazy, dehazy, ale = record['x'], record['y'], record['ale']
        hazys += [h for h in hazy]
        dehazys += [deh for deh in dehazy]

    hazys = tf.stack(hazys, axis=0)
    dehazys = tf.stack(dehazys, axis=0)

    hazy_labels, dehazy_labels = tf.ones(shape=tf.shape(hazys)[0]), tf.zeros(shape=tf.shape(dehazys)[0])

    images = tf.concat([hazys, dehazys], axis=0)
    labels = tf.concat([hazy_labels, dehazy_labels], axis=0)

    indices = tf.range(tf.shape(images)[0])
    indices = tf.random.shuffle(indices)
    images = tf.gather(images, indices)
    labels = tf.gather(labels, indices)

    return images, labels

def create_global_dicriminator(input_channels, norm_type='batchnorm', target=True):
    """ PatchGAN discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
        input_channels: Input channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        target: Bool, indicating whether target image is an input or not.

    Returns:
        Discriminator model
    """

    patch_gan = pix2pix.discriminator(input_channels, norm_type=norm_type, target=target)

    inputs = tf.keras.layers.Input(shape=[height, width, input_channels])
    d_output = patch_gan(inputs)
    d_flatten = tf.keras.layers.Flatten()(d_output)
    last = tf.keras.layers.Dense(2)(d_flatten)

    return tf.keras.Model(inputs=inputs, outputs=last)

def test():
    dataset = load_dataset(tfrecords_name)

    global_discriminator = create_global_dicriminator(3, norm_type='instancenorm', target=False)
    optimizer = tf.keras.optimizers.Adam(1e-5)

    ckpt = tf.train.Checkpoint(
        global_discriminator=global_discriminator, 
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    accuracy = tf.keras.metrics.Accuracy()

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored.')

    # for record in dataset:
    #     images, labels = generate_test_set(record)
    #     print(images.shape, labels.shape)

    #     class_names = ['Dehazy', 'Hazy']

    #     logits = global_discriminator(images, training=False)
    #     preds = tf.argmax(logits)
    #     print(preds.shape)

    #     num_images = tf.shape(images)[0].numpy()
    #     for i in range(num_images):
    #         plt.subplot(1, num_images, i + 1)
    #         plt.imshow(images[i] * 0.5 + 0.5)
    #         plt.title('Prediction: {}, Label: {}'.format(class_names[preds[i]], class_names[labels[i]]))

    #     plt.show()

    with tf.device('/device:cpu:0'):
        images, labels = generate_test_set(dataset)
    
        for image, label in zip(images, labels):
            with tf.device('/device:gpu:0'):
                logits = global_discriminator(image[tf.newaxis, ...], training=False)
            preds = tf.argmax(logits)
            print(preds.numpy())

if __name__ == "__main__":
    test()
