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


def preprocess(image, label):
    import tensorflow as tf

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # converting dtype changes uint8 [0..255] to float [0.,1.]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = (image - tf.reshape(mean, [1, 1, 3])) / tf.reshape(std, [1, 1, 3])

    label = tf.one_hot(label, depth=120, dtype=tf.int32)

    return image, label


def augment(image, label):
    import tensorflow as tf
    import tensorflow_addons as tfa

    pad = 2
    # random crop with zero-padding
    image = tf.image.resize_with_crop_or_pad(image,
                                             16 + pad * 2,
                                             16 + pad * 2)
    image = tf.image.random_crop(image, size=[16, 16, 3])
    # random LR flip
    image = tf.image.random_flip_left_right(image)
    # cutout
    image = tfa.image.random_cutout(tf.expand_dims(image, 0), (4, 4))
    image = tf.squeeze(image, axis=0)
    return image, label


def load_data():
    def load_ds(split):
        import tensorflow_datasets as tfds

        ds = tfds.load('imagenet_resized/16x16', as_supervised=True,
                       split=split)
        return ds

    def load_ds_train():
        import tensorflow as tf

        ds_train = load_ds('train')
        ds_train = (
            ds_train
                .filter(lambda x, y: y < 120)  # ImageNet-16-120
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        )
        return ds_train

    def load_ds_valid():
        import tensorflow as tf

        ds_valid = load_ds('validation')
        ds_valid = (
            ds_valid
                .filter(lambda x, y: y < 120)  # ImageNet-16-120
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
        )
        return ds_valid

    return {
        'train_gen': load_ds_train,
        'train_size': 151700,
        'valid_gen': load_ds_valid,
        'valid_size': 6000,
        'types': ({'input_0': 'float32'}, 'int32'),
        'shapes': ({'input_0': (16, 16, 3)}, (120,)),
    }
