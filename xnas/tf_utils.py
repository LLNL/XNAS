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
from xnas import utils

logger = utils.get_logger(__name__)


def load_model(*args, custom_objects=None, **kwargs):
    import tensorflow as tf
    from xnas.search_space_common import TransposeLayer

    custom_objects = custom_objects or {}
    custom_objects['TransposeLayer'] = TransposeLayer
    return tf.keras.models.load_model(
        *args, custom_objects=custom_objects, **kwargs)


def keras_logger_callback(config):
    import tensorflow as tf

    class LoggingCallback(tf.keras.callbacks.Callback):

        def __init__(self):
            id_ = 'ID=' + str(config.get('id', '?'))
            self._logger = utils.get_logger(id_)
            self._epoch = '<<< UNDEFINED >>>'  # this should never be printed
            super().__init__()

        def _log_logs(self, logs):
            if not logs:
                return
            maxlen = max(len(k) for k in logs)
            for k, v in logs.items():
                if k in {'outputs'}:  # skip these keys
                    continue
                self._logger.info(f'  {k:>{maxlen}} = {v}')

        def on_train_begin(self, logs=None):
            self._logger.info('!!! Starting training !!!')
            self._log_logs(logs)

        def on_train_end(self, logs=None):
            self._logger.info('!!! Stopped training !!!')
            self._log_logs(logs)

        def on_epoch_begin(self, epoch, logs=None):
            logger.info(f'Start epoch {epoch} of training.')
            self._log_logs(logs)
            self._epoch = epoch

        def on_epoch_end(self, epoch, logs=None):
            logger.info(f'End epoch {epoch} of training.')
            self._log_logs(logs)

        def on_test_begin(self, logs=None):
            self._logger.info('!!! Starting testing !!!')
            self._log_logs(logs)

        def on_test_end(self, logs=None):
            self._logger.info('!!! Stopped testing !!!')
            self._log_logs(logs)

        def on_predict_begin(self, logs=None):
            self._logger.info('!!! Starting predicting !!!')
            self._log_logs(logs)

        def on_predict_end(self, logs=None):
            self._logger.info('!!! Stopped prediction !!!')
            self._log_logs(logs)

        def on_train_batch_begin(self, batch, logs=None):
            self._logger.info(f'...Training: start of batch {batch} (epoch '
                              f'{self._epoch})')
            self._log_logs(logs)

        def on_train_batch_end(self, batch, logs=None):
            self._logger.info(f'...Training: end of batch {batch} (epoch '
                              f'{self._epoch})')
            self._log_logs(logs)

        def on_test_batch_begin(self, batch, logs=None):
            self._logger.info(f'...Evaluating: start of batch {batch}')
            self._log_logs(logs)

        def on_test_batch_end(self, batch, logs=None):
            self._logger.info(f'...Evaluating: end of batch {batch}')
            self._log_logs(logs)

        def on_predict_batch_begin(self, batch, logs=None):
            self._logger.info(f'...Prediction: start of batch {batch}')
            self._log_logs(logs)

        def on_predict_batch_end(self, batch, logs=None):
            self._logger.info(f'...Prediction: end of batch {batch}')
            self._log_logs(logs)

    return LoggingCallback()
