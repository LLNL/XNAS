"""
MIT License

Copyright (c) 2022, Lawrence Livermore National Security, LLC
Written by Zachariah Carmichael et al.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from xnas.utils import get_logger

logger = get_logger(__name__)


def preprocess(image, label):
    import tensorflow as tf

    mean = [0.13066044]
    std = [0.3081079]

    # converting dtype changes uint8 [0..255] to float [0.,1.]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = (image - tf.reshape(mean, [1, 1, 1])) / tf.reshape(std, [1, 1, 1])

    label = tf.one_hot(label, depth=10, dtype=tf.int32)

    return image, label


def augment(image, label):
    import tensorflow as tf
    import tensorflow_addons as tfa

    pad = 4
    # random crop with zero-padding
    image = tf.image.resize_with_crop_or_pad(image,
                                             28 + pad * 2,
                                             28 + pad * 2)
    image = tf.image.random_crop(image, size=[28, 28, 1])
    # random LR flip
    image = tf.image.random_flip_left_right(image)
    # cutout
    image = tfa.image.random_cutout(tf.expand_dims(image, 0), (8, 8))
    image = tf.squeeze(image, axis=0)
    return image, label


def load_data():
    def load_train():
        import tensorflow as tf
        import tensorflow_datasets as tfds

        ds_train = tfds.load('mnist', as_supervised=True, split='train')
        ds_train = (
            ds_train
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        )
        return ds_train

    def load_test():
        import tensorflow as tf
        import tensorflow_datasets as tfds

        ds_test = tfds.load('mnist', as_supervised=True, split='test')
        ds_test = (
            ds_test
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
        )
        return ds_test

    train_size, valid_size = 60000, 10000

    return {
        'train_gen': load_train,
        'train_size': train_size,
        'valid_gen': load_test,
        'valid_size': valid_size,
        'types': ({'input_0': 'float32'}, 'int32'),
        'shapes': ({'input_0': (28, 28, 1)}, (10,)),
    }
