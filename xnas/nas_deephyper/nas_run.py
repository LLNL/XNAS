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

-------------------------------------------------------------------------------

Based off of `deephyper.nas.run.alpha.run` function: used to evaluate a deep
neural network by loading the data, building the model, training the model
and returning a scalar value corresponding to the objective defined in the used
:class:`deephyper.problem.NaProblem`.
"""
import logging

import numpy as np
import scipy.spatial
import scipy.stats

from xnas.utils import get_logger
from xnas.utils import log_exceptions
from xnas.utils import configure_logging

configure_logging(level=logging.INFO)
logger = get_logger(__name__)


def nvidia_smi():
    import subprocess

    result = subprocess.run(['nvidia-smi'], capture_output=True)
    return result.stdout.decode('UTF-8')


class DisconnectedGraphError(AssertionError):
    pass


@log_exceptions(logger=logger)
def run(config):
    """
    Isolate run process to its own process due to VRAM issues with TensorFlow:
    https://github.com/tensorflow/tensorflow/issues/36465#issuecomment-582749350

    Args:
        config: DeepHyper NAS configuration dict

    Returns:
        score (float)
    """
    import multiprocessing as mp
    import shutil
    import os
    import sys

    config['verbose'] = 1

    configure_logging(level=logging.INFO)
    logger.info('Starting XNAS Run function')
    logger.info(f'nvidia-smi (pre-run) TF={"tensorflow" in sys.modules}'
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}'
                f'\n{nvidia_smi()}')
    mp.set_start_method('spawn', force=True)

    # _run_process will populate the queue with its return value (score)
    score = _run_process(config)

    logger.info(f'nvidia-smi (post-run) TF={"tensorflow" in sys.modules} '
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}'
                f'\n{nvidia_smi()}')

    # check if XAI requested
    xai_flag = config.get('multiobjective_explainability', False)
    xai_type = config.get('explainability_type', 'activations').lower()
    logger.info(f'XAI flag multiobjective_explainability is {xai_flag}')
    if xai_flag:
        logger.info('Using result from NAS run to compute introspectability')
        # unpack
        score, model_path_tf, ds_val = score

        if model_path_tf is None:
            # model failed to be created, so nothing to do on XAI side
            # not applicable (not implemented for architecture!)
            logger.info('TF model path is None!')
            xai_layerwise = None
            xai_fitness = None
        else:
            logger.info(
                f'nvidia-smi (pre-xai) TF={"tensorflow" in sys.modules}'
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}'
                f'\n{nvidia_smi()}'
            )
            try:
                if xai_type in {'activations', 'activations-imagenet'}:
                    logger.info('XAI Fitness: Running activations')

                    logits, activations = _run_activations(
                        model_path_tf, ds_val, config=config)

                    # save activations to file
                    save_dir = os.path.join(config.get('log_dir', ''),
                                            'activations')
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    save_path = os.path.join(save_dir,
                                             str(config['id']) + '.npz')
                    logger.info(f'Saving activations to {save_path}')
                    np.savez_compressed(
                        save_path,
                        **{str(k): v for k, v in activations.items()}
                    )

                    xai_layerwise = None
                    xai_fitness = introspection_score_classification(
                        logits, activations,
                        imagenet=(xai_type == 'activations-imagenet'))
                else:
                    raise ValueError(f'Unknown XAI fitness metric name: '
                                     f'{xai_type}')
            finally:
                # clean up TF model
                shutil.rmtree(os.path.dirname(model_path_tf))
            logger.info(
                f'nvidia-smi (post-xai) TF={"tensorflow" in sys.modules}'
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}'
                f'\n{nvidia_smi()}')

        return {
            'score': score,
            'xai_fitness': xai_fitness,
            'xai_layerwise': xai_layerwise
        }
    else:
        return score


_IMAGENET_DF = None


def imagenet_similarity_score(class_a, class_b, metric='path_sim'):
    global _IMAGENET_DF

    import pathlib
    import pandas as pd
    import xnas

    # classes should be integers
    if class_a > class_b:
        class_a, class_b = class_b, class_a
    else:
        assert class_a != class_b, (
            f'Similarity of the same class ({class_a}), do not do that')

    if _IMAGENET_DF is None:
        dirname = pathlib.Path(xnas.__file__).parent.parent
        filename = ('imagenet_resized_label_distances_'
                    '9aca0f6d9df6da4fa8c2f33ee7fb8fd9.csv')
        path = dirname / filename
        _IMAGENET_DF = df = pd.read_csv(path, index_col=['idx_i', 'idx_j'])
    else:
        df = _IMAGENET_DF
    return df.loc[(class_a, class_b), metric]


def introspection_score_classification(logits, xai_layerwise_maps,
                                       imagenet=False):
    """introspection score ~= XAI fitness"""
    # form of input if already aggregated by another process (likely for memory
    #  efficiency)
    input_is_agg = isinstance(xai_layerwise_maps, dict)
    if logits is not None or input_is_agg:
        # Compute XAI score
        if input_is_agg:
            xai_layerwise_maps: dict
            assert logits is None, 'do not give me logits if input_is_agg!!!'
            if not xai_layerwise_maps:
                logger.warning('XAI layerwise maps empty! fitness is None')
                return None
            agg_classes, agg_maps = zip(*xai_layerwise_maps.items())
        else:
            # This is classification only:
            assert logits.ndim == 2, logits.shape
            assert len(logits) == len(xai_layerwise_maps), (
                len(logits), len(xai_layerwise_maps))

            predictions = np.argmax(logits, axis=1)

            sort_idxs = np.argsort(predictions)
            predictions_sorted = predictions[sort_idxs]
            xai_layerwise_maps = xai_layerwise_maps[sort_idxs]

            class_ends = np.where(
                predictions_sorted[:-1] != predictions_sorted[1:])[0] + 1
            class_ends = np.concatenate(
                [class_ends, [len(predictions_sorted)]])

            class_start = 0
            agg_maps = []
            agg_classes = []
            for class_end in class_ends:
                agg_classes.append(predictions_sorted[class_start])

                xai_layerwise_maps_class = \
                    xai_layerwise_maps[class_start:class_end]
                xai_layerwise_map_class = \
                    xai_layerwise_maps_class.mean(axis=0)
                agg_maps.append(xai_layerwise_map_class)
                class_start = class_end

        # Increase precision for distance comp.
        agg_maps = np.asarray(agg_maps).astype('float64')
        # ideally this term is maximized:
        xai_fitness = scipy.spatial.distance.pdist(
            agg_maps.reshape(len(agg_maps), -1),
            metric='cosine',
        )

        if imagenet:
            logger.info('ImageNet weighting will be incorporated!')

            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            m = len(agg_maps)
            for i in range(m):
                for j in range(i + 1, m):
                    # location in xai_fitness that (i, j) dist is stored
                    idx = m * i + j - ((i + 2) * (i + 1)) // 2
                    # map indices i & j to original class index using
                    #  agg_classes
                    xai_fitness[idx] *= imagenet_similarity_score(
                        agg_classes[i], agg_classes[j])

        xai_nan = np.isnan(xai_fitness)
        if xai_nan.all():
            logger.warning('All values of xai_fitness are NaN!')
            xai_fitness = -np.inf
        elif xai_nan.any():
            logger.warning(f'Some values of xai_fitness are NaN! '
                           f'Dropping NaNs...')
            xai_fitness = np.mean(xai_fitness[~xai_nan])
        else:
            xai_fitness = xai_fitness.mean()
    else:
        xai_fitness = None
    return xai_fitness


def layer_of_interest(name):
    import re

    return bool(
        re.search(r'(^|/)((separable|depthwise)?conv|activation)', name))


@log_exceptions(logger=logger)
def _run_activations(
        model_dir_tf,
        dataset,
        config,
        n_max=25000,
):
    configure_logging(level=logging.INFO)
    logger.info('Enter run activations')

    from math import ceil
    import tensorflow as tf
    from xnas.tf_utils import load_model

    if isinstance(dataset, list) and len(dataset) == 1:
        dataset = dataset[0]

    if isinstance(dataset, tf.data.Dataset):
        num_epochs = config['hyperparameters']['num_epochs']
        batch_size = config['hyperparameters']['batch_size']
        # approx actual size
        try:
            len(dataset)
        except TypeError:
            logger.warning('Dataset has unknown length, cannot downsample')
        else:
            actual_size = len(dataset) * batch_size / num_epochs
            if actual_size > n_max:
                dataset = dataset.take(ceil(n_max / batch_size))
            else:
                # just take one epoch's worth
                dataset = dataset.take(ceil(len(dataset) / num_epochs))
    elif len(dataset) > n_max:
        # numpy dataset
        dataset = np.random.choice(dataset, size=n_max, replace=False)

    logger.info('Load TF model')
    model: tf.keras.Model = load_model(model_dir_tf)
    outputs = [layer.output for layer in model.layers
               if layer_of_interest(layer.name)]
    if len(outputs) == 0:
        logger.warning(f'NO ACTIVATIONS FOR ID: {config["id"]} '
                       f'({config["arch_seq"]})')
        return None, {}

    outputs.append(model.output)
    logger.info('Create get_activations model')
    get_activations = tf.keras.Model([model.input], outputs)
    logger.info('Get activations and logits')

    class_activations_map = {}
    n = 0
    for batch_i, _ in dataset:  # data, labels
        if isinstance(batch_i, (list, tuple)):
            assert len(batch_i) == 1, len(batch_i)
            batch_i = batch_i[0]
        if isinstance(batch_i, dict):
            assert len(batch_i) == 1, [*batch_i.keys()]
            batch_i = batch_i['input_0']
        n += len(batch_i)
        # model() supposedly fixes problem with model.predict() retracing
        # https://stackoverflow.com/q/66271988/6557588
        # https://github.com/tensorflow/tensorflow/issues/34025
        *activations_i, logits_i = get_activations(batch_i)

        logits_i = logits_i.numpy()
        assert logits_i.ndim == 2, logits_i.shape
        predictions_i = np.argmax(logits_i, axis=1).astype(int)
        del logits_i
        activations_i = np.concatenate([
            act.numpy().reshape(len(batch_i), -1) for act in activations_i],
            axis=1
        )
        assert len(predictions_i) == len(activations_i), (
            predictions_i.shape, activations_i.shape)

        for prediction_i, activation_i in zip(predictions_i, activations_i):
            if prediction_i in class_activations_map:
                class_activations_map[prediction_i] += activation_i
            else:
                class_activations_map[prediction_i] = activation_i
        del predictions_i
        del activations_i
    for class_i in class_activations_map:
        # divide by total number of samples
        class_activations_map[class_i] /= n

    logits = None  # keep None so other classes know input is aggregated
    return logits, class_activations_map


def _parse_objective(objective):
    """
    Modified from deephyper/nas/run/util.py function compute_objective
    """
    if isinstance(objective, str):
        negate = (objective[0] == '-')
        if negate:
            objective = objective[1:]

        split_objective = objective.split('__')
        kind = split_objective[1] if len(split_objective) > 1 else 'last'
        mname = split_objective[0]
        # kind: min/max/last
        if negate:
            if kind == 'min':
                kind = 'max'
            elif kind == 'max':
                kind = 'min'
        return mname, kind
    elif callable(objective):
        logger.warn('objective is a callable, not a str, setting kind="last"')
        return None, 'last'
    else:
        raise TypeError(f'unknown objective type {type(objective)}')


@log_exceptions(logger=logger)
def _run_process(config):
    import os
    import gc

    # https://stackoverflow.com/questions/1367373/python-subprocess-popen-oserror-errno-12-cannot-allocate-memory
    gc.collect()

    from xnas.nas_deephyper.monkey_patch import (
        monkey_patch__TrainerTrainValid)

    TrainerTrainValid = monkey_patch__TrainerTrainValid(logger)

    import traceback
    import tempfile
    import shutil
    from textwrap import dedent
    from typing import Iterable

    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa

    if not hasattr(tf.data, 'AUTOTUNE'):  # normalize name
        tf.data.AUTOTUNE = tf.data.experimental.AUTOTUNE

    from deephyper.contrib.callbacks import import_callback
    from deephyper.nas.run.util import (
        compute_objective,
        load_config,
        preproc_trainer,
        setup_data,
        setup_search_space,
        default_callbacks_config,
        HistorySaver,
    )
    from xnas.tf_utils import keras_logger_callback

    configure_logging(level=logging.INFO)

    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True)

    # register TFA to keras
    tfa.register_all(custom_kernels=False)

    # setup history saver
    if 'log_dir' in config and config['log_dir'] is None:
        config['log_dir'] = ''

    save_dir = os.path.join(config.get('log_dir', ''), 'save')
    saver = HistorySaver(config, save_dir)
    saver.write_config()
    saver.write_model(None)

    # environment information
    logger.info(dedent(f'''
        ---------------------------------
             Environment Information
        ---------------------------------
        SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')}
        CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}
        tf.config.list_physical_devices('GPU')={tf.config.list_physical_devices('GPU')}
        tf.test.is_built_with_cuda()={tf.test.is_built_with_cuda()}
        tf.test.is_built_with_gpu_support()={tf.test.is_built_with_gpu_support()}
        tf.test.is_gpu_available()={tf.test.is_gpu_available()}
        tf.config.experimental.get_visible_devices()={tf.config.experimental.get_visible_devices()}
        ---------------------------------\
        '''))

    # GPU Configuration if available
    logger.info('GPU configuration')
    physical_devices = tf.config.list_physical_devices('GPU')
    for i in range(len(physical_devices)):
        device_i = physical_devices[i]
        if tf.config.experimental.get_memory_growth(device_i):
            logger.info(f'memory growth already True for device {device_i}')
            continue
        try:
            tf.config.experimental.set_memory_growth(device_i, True)
        except:  # noqa
            # Invalid device or cannot modify virtual devices once initialized.
            logger.warning(f'error memory growth for GPU device {device_i}')
            logger.warning(traceback.format_exc())

    # Threading configuration
    logger.info('Threading configuration')
    if (len(physical_devices) == 0 and
            os.environ.get('OMP_NUM_THREADS') is not None):
        logger.info(f'OMP_NUM_THREADS is {os.environ.get("OMP_NUM_THREADS")}')
        num_intra = int(os.environ.get('OMP_NUM_THREADS'))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(num_intra)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:  # Session already initialized
            pass
        tf.config.set_soft_device_placement(True)

    seed = config['seed']
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    load_config(config)

    # Explainability and multi-objective config options
    xai_flag = config.get('multiobjective_explainability', False)

    logger.info('Setting up data')
    input_shape, output_shape = setup_data(config)

    search_space = setup_search_space(config, input_shape, output_shape,
                                      seed=seed)

    try:
        logger.info('Creating model')
        output_tensor = search_space.create_tensor_aux(
            search_space.graph, search_space.output_node)
        if isinstance(output_tensor, list) and len(output_tensor) == 0:
            # This will happen when there is no path between the input and
            #  output, e.g., all zeroize operations are selected in a NAS 201
            #  cell
            raise DisconnectedGraphError(
                'The graph is disconnected! This is likely OK unless this '
                'happens for any combination of operations within the search '
                'space.'
            )
        model = search_space.create_model()
    except:  # noqa
        logger.warning('Error: Model creation failed...')
        logger.warning(traceback.format_exc())
        # penalising actions if model cannot be created
        result = -float('inf')
        if xai_flag:
            result = (result, None, None)
        logger.warning('Model could not be created returning -Inf!')
    else:
        # Setup callbacks
        callbacks = [keras_logger_callback(config),
                     tf.keras.callbacks.TerminateOnNaN()]
        ckpt_dir = None
        ckpt_filepath = None

        cb_requires_valid = False  # Callbacks requires validation data
        callbacks_config = config['hyperparameters'].get('callbacks')
        if callbacks_config is None:
            callbacks_config = {}
        callback_mc = None
        if xai_flag:
            mname, mkind = _parse_objective(config['objective'])
            if mkind != 'last':
                logger.info('Creating model checkpoint callback!!!')
                ckpt_dir = tempfile.mkdtemp(
                    prefix='deephyper_objective_model_weights_xai_')
                ckpt_filepath = os.path.join(ckpt_dir, 'checkpoint')

                # save the weights of the best model (according to specified
                #  objective) to restore later in XAI evaluation.
                callback_mc = tf.keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_filepath,
                    monitor=mname,
                    save_best_only=True,
                    save_weights_only=True,
                    save_freq='epoch',
                    mode=mkind,  # mkind will be either min or max here
                )

        if callbacks_config:
            logger.info('Adding callbacks')
            for cb_name, cb_conf in callbacks_config.items():
                if cb_name in default_callbacks_config:
                    for k, v in default_callbacks_config[cb_name].items():
                        cb_conf.setdefault(k, v)

                    # Special dynamic parameters for callbacks
                    if cb_name == 'ModelCheckpoint':
                        default_callbacks_config[cb_name]['filepath'] = (
                            saver.model_path)

                    # replace patience hyperparameter
                    if 'patience' in default_callbacks_config[cb_name]:
                        patience = config['hyperparameters'].get(
                            f'patience_{cb_name}')
                        if patience is not None:
                            default_callbacks_config[cb_name]['patience'] = (
                                patience)

                    # Import and create corresponding callback
                    Callback = import_callback(cb_name)
                    callbacks.append(
                        Callback(**default_callbacks_config[cb_name]))  # noqa

                    if cb_name in ['EarlyStopping']:
                        cb_requires_valid = (
                                'val' in cb_conf['monitor'].split('_'))
                else:
                    logger.error(f'\'{cb_name}\' is not an accepted callback!')

        logger.info('Create TrainerTrainValid')
        trainer = TrainerTrainValid(config=config, model=model)
        trainer.callbacks.extend(callbacks)
        if callback_mc is not None:
            trainer.callbacks.append(callback_mc)

        last_only, with_pred = preproc_trainer(config)
        last_only = last_only and not cb_requires_valid

        logger.info('Start training')
        history = trainer.train(with_pred=with_pred, last_only=last_only)

        # save history
        saver.write_history(history)
        # log history
        for mname, mvals in history.items():
            if not isinstance(mvals, Iterable):
                mvals = (mvals,)
            logger.info(
                f'History Stats {mname}\n'
                f'     min={min(mvals)}\n'
                f'     max={max(mvals)}\n'
                f'    last={mvals[-1]}'
            )

        logger.info('Computing objective')
        result = compute_objective(config['objective'], history)

        if xai_flag:
            logger.info('Objective computed - XAI flag is set')
            # XAI Time
            if ckpt_filepath is not None:
                logger.info(
                    'Restoring best weights by user-specified objective')
                # restore best weights found in training as judged by objective
                #  otherwise current weights will be used
                model.load_weights(ckpt_filepath)
            # save model to disk and return path in queue
            model_dir = tempfile.mkdtemp(
                prefix='deephyper_model_tf_transfer_xai_')
            model_path = os.path.join(model_dir, 'tf-keras-model.h5')
            logger.info('Save model with best weights (tf.keras)')
            tf.keras.models.save_model(
                model, model_path, include_optimizer=False, save_format='h5')
            result = (result, model_path, trainer.dataset_valid)

        # clean up temporary files
        if ckpt_dir is not None and os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

    if np.isnan(result[0] if xai_flag else result):
        logger.info('Computed objective is NaN returning -Inf instead!')
        if xai_flag:
            result = (-float('inf'),) + result[1:]
        else:
            result = -float('inf')

    gc.collect()
    return result
