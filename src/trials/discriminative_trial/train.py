# -*- coding: utf-8 -*-

import tensorflow as tf
import pix2pix

tfrecords_name = '../../datasets/RICE1.tfrecords'
checkpoint_path = 'checkpoints/train/'
batch_size = 2
height = 512
width = 512

buffer_size = batch_size * 4

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

def loss_func(logits, labels):
    return loss_obj(y_true=labels, y_pred=logits)

def train_loop():
    dataset = load_dataset(tfrecords_name)

    global_discriminator = create_global_dicriminator(3, norm_type='instancenorm', target=False)
    optimizer = tf.keras.optimizers.Adam(1e-5)

    ckpt = tf.train.Checkpoint(
        global_discriminator=global_discriminator, 
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    accuracy = tf.keras.metrics.Accuracy()

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = global_discriminator(inputs, training=True)
            loss = loss_func(logits, labels)
        gradients = tape.gradient(loss, global_discriminator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, global_discriminator.trainable_variables))
        return loss

    step = 0
    for epoch in range(100):
        dataset = dataset.shuffle(buffer_size=buffer_size)
        train_set = dataset.take(400 // batch_size)
        val_set = dataset.skip(400 // batch_size)

        print('Training')
        for record in train_set:
            step += 1
            hazy, dehazy, ale = record['x'], record['y'], record['ale']
            hazy_label, dehazy_label = tf.ones(shape=[batch_size, ]), tf.zeros(shape=[batch_size, ])
            inputs = tf.concat([hazy, dehazy], axis=0)
            labels = tf.concat([hazy_label, dehazy_label], axis=0)
            loss = train_step(inputs, labels)
            print('Step {} Training loss:{}'.format(step, loss.numpy()))

        ckpt_save_path = ckpt_manager.save()
        print('Checkpoint for epoch {} saved at {}'.format(epoch + 1, ckpt_save_path))

        n = 0
        print('Validating')
        for record in val_set:
            hazy, dehazy, ale = record['x'], record['y'], record['ale']
            hazy_label, dehazy_label = tf.ones(shape=[batch_size, ]), tf.zeros(shape=[batch_size, ])
            inputs = tf.concat([hazy, dehazy], axis=0)
            labels = tf.concat([hazy_label, dehazy_label], axis=0)

            logits = global_discriminator(inputs, training=False)
            preds = tf.argmax(logits, axis=-1)
            accuracy.update_state(labels, preds)
        print('Validation accuracy: {}'.format(accuracy.result().numpy()))
        accuracy.reset_states()

if __name__ == "__main__":
    train_loop()