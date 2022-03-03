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


def monkey_patch__TrainerTrainValid(logger):  # noqa
    # monkey patch the shuffle buffer size...
    # https://github.com/deephyper/deephyper/blob/c7608e0c61bd805c109145744b567cbb6cf01673/deephyper/nas/trainer/train_valid.py
    from deephyper.nas.trainer.train_valid import TrainerTrainValid

    def make_dataset(data_config_type, X, Y, gen, data_types, data_shapes,
                     size, batch_size, num_epochs):
        import tensorflow as tf

        if data_config_type == "ndarray":
            if type(Y) is list:
                output_mapping = {f"output_{i}": tY for i, tY in
                                  enumerate(Y)}
            else:
                output_mapping = Y
            dataset = tf.data.Dataset.from_tensor_slices(
                ({f"input_{i}": tX for i, tX in enumerate(X)},
                 output_mapping)
            ).cache()
        else:
            maybe_dataset = gen()
            if isinstance(maybe_dataset, tf.data.Dataset):
                logger.info('Using gen() as dataset! Ensure you called '
                            'cache()')
                dataset = maybe_dataset
            else:
                dataset = tf.data.Dataset.from_generator(
                    lambda: maybe_dataset,
                    output_types=data_types,
                    output_shapes=(
                        {
                            f"input_{i}": tf.TensorShape(
                                [*data_shapes[0][f"input_{i}"]])
                            for i in range(len(data_shapes[0]))
                        },
                        tf.TensorShape([*data_shapes[1]]),
                    ),
                ).cache()
        dataset = (
            dataset
                .shuffle(min(size, max(batch_size * 10, 1000)),
                         reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
                .repeat(num_epochs)
        )
        logger.info(
            f'DS made: batch_size={batch_size} and num_epochs={num_epochs}')
        return dataset

    def set_dataset_train(self):
        self.dataset_train = make_dataset(
            data_config_type=self.data_config_type,
            X=getattr(self, 'train_X', None),
            Y=getattr(self, 'train_Y', None),
            gen=getattr(self, 'train_gen', None),
            data_types=getattr(self, 'data_types', None),
            data_shapes=getattr(self, 'data_shapes', None),
            size=self.train_size,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
        )

    def set_dataset_valid(self):
        self.dataset_valid = make_dataset(
            data_config_type=self.data_config_type,
            X=getattr(self, 'valid_X', None),
            Y=getattr(self, 'valid_Y', None),
            gen=getattr(self, 'valid_gen', None),
            data_types=getattr(self, 'data_types', None),
            data_shapes=getattr(self, 'data_shapes', None),
            size=self.valid_size,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
        )

    def model_compile(self: TrainerTrainValid):
        from inspect import signature
        from math import ceil
        from deephyper.nas import train_utils as U  # noqa
        from deephyper.core.exceptions import DeephyperRuntimeError
        import tensorflow as tf

        try:
            optimizer_fn = tf.keras.optimizers.get(
                self.optimizer_name).__class__
        except ValueError:
            try:
                optimizer_fn = tf.keras.utils.get_custom_objects()[
                    f'Addons>{self.optimizer_name.upper()}']
            except KeyError:
                optimizer_fn = U.selectOptimizer_keras(self.optimizer_name)

        opti_parameters = signature(optimizer_fn).parameters
        params = {}

        learning_rate = self.learning_rate
        if 'LearningRateScheduler' not in self.config_hp.get('callbacks', {}):
            logger.info('Adding Cosine Annealing LR Schedule!!')

            initial_learning_rate = learning_rate
            decay_steps = ceil((self.train_size / self.batch_size) *
                               self.num_epochs)
            try:
                CosineDecay = tf.keras.optimizers.schedules.CosineDecay
            except AttributeError:
                CosineDecay = tf.keras.experimental.CosineDecay
            LearningRateSchedule = (
                tf.keras.optimizers.schedules.LearningRateSchedule)

            warmup_epochs = 0.25
            warmup_steps = ceil(
                (self.train_size / self.batch_size) * warmup_epochs)

            logger.info(f'CosineDecay with initial_learning_rate='
                        f'{initial_learning_rate}, decay_steps={decay_steps},'
                        f'warmup_steps={warmup_steps}, alpha=0.0')

            lr_schedule = CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps - warmup_steps,
                alpha=0.0,
            )

            class CompleteLRSchedule(LearningRateSchedule):
                """combined warmup --> schedule"""

                def get_config(self):
                    return super().get_config()

                @tf.function
                def __call__(self, step):
                    if step < warmup_steps:
                        return initial_learning_rate * step / warmup_steps
                    else:
                        # cosine
                        return lr_schedule(step)

            learning_rate = CompleteLRSchedule()

        if "lr" in opti_parameters:
            params["lr"] = learning_rate
        elif "learning_rate" in opti_parameters:
            params["learning_rate"] = learning_rate
        else:
            raise DeephyperRuntimeError(
                f"The learning_rate parameter is not found among optimizer "
                f"arguments: {opti_parameters}"
            )

        if "epsilon" in opti_parameters:
            params["epsilon"] = self.optimizer_eps

        for hparam in ['momentum', 'weight_decay', 'nesterov']:
            if hparam in self.config_hp:
                if hparam == 'weight_decay':
                    if hparam in opti_parameters:
                        logger.info('Using decoupled weight decay!')
                        params[hparam] = self.config_hp[hparam]
                    else:
                        logger.info('Using L2 global weight decay!')
                        # https://stackoverflow.com/questions/41260042/global-weight-decay-in-keras/54564848
                        for layer in self.model.layers:
                            if isinstance(layer, (tf.keras.layers.Conv2D,
                                                  tf.keras.layers.Dense)):
                                layer.add_loss(
                                    lambda layer=layer, hparam=hparam:  # noqa
                                    tf.keras.regularizers.l2(
                                        self.config_hp[hparam])(layer.kernel)
                                )
                            if (hasattr(layer, 'bias_regularizer') and
                                    layer.use_bias):
                                layer.add_loss(
                                    lambda layer=layer, hparam=hparam:  # noqa
                                    tf.keras.regularizers.l2(
                                        self.config_hp[hparam])(layer.bias)
                                )
                else:
                    params[hparam] = self.config_hp[hparam]

        logger.info(f'Instantiating optimizer {optimizer_fn} with '
                    f'hparams:\n{params}')

        self.optimizer = optimizer_fn(**params)

        if type(self.loss_metrics) is dict:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_metrics,
                loss_weights=self.loss_weights,
                metrics=self.metrics_name,
            )
        else:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_metrics,
                metrics=self.metrics_name,
            )

    logger.info('Monkey-patching TrainerTrainValid.set_dataset_train')
    TrainerTrainValid.set_dataset_train = set_dataset_train
    logger.info('Monkey-patching TrainerTrainValid.set_dataset_valid')
    TrainerTrainValid.set_dataset_valid = set_dataset_valid
    logger.info('Monkey-patching TrainerTrainValid.model_compile')
    TrainerTrainValid.model_compile = model_compile
    return TrainerTrainValid
