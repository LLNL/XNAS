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
from functools import partial

from abc import ABC
from abc import abstractmethod

import numpy as np
import scipy.stats

import tensorflow as tf

from deephyper.nas.space import AutoKSearchSpace
from deephyper.nas.space.node import VariableNode
from deephyper.nas.space.node import ConstantNode
from deephyper.nas.space.node import MimeNode

from deephyper.nas.space.op.basic import Operation
from deephyper.nas.space.op.basic import Tensor

from deephyper.nas.space.op.cnn import Conv2D
from deephyper.nas.space.op.cnn import AvgPool2D
from deephyper.nas.space.op.cnn import MaxPool2D
from deephyper.nas.space.op.cnn import SeparableConv2D

from deephyper.nas.space.op.op1d import Dense
from deephyper.nas.space.op.op1d import BatchNormalization
from deephyper.nas.space.op.op1d import Dropout
from deephyper.nas.space.op.op1d import Identity

from deephyper.nas.space.op.merge import AddByProjecting

from xnas.utils import get_logger

logger = get_logger(__name__)


def summarize_search_space_arch(search_space, ops, plot_shapes=True):
    """Generate a random neural network from the search_space definition"""
    from tensorflow.keras.utils import plot_model

    search_space.set_ops(ops)

    print(f'This search_space needs {len(ops)} choices to generate a neural '
          'network.')

    model = search_space.create_model()
    model.summary()

    print('ops:')
    print(ops)
    print()
    for n in search_space.nodes:
        print(n.op)
        print(vars(n.op))
    print()

    print(f'The size of the search space is {search_space.size}')

    plot_model(model, to_file='sampled_neural_network.png',
               show_shapes=plot_shapes)
    print('The sampled_neural_network.png file has been generated.')


class BlockOps:
    def __init__(self, dropout=None, batchnorm=True, scope=None):
        self.scope = scope

        self._dropout = dropout
        self._dropout_layer = None

        self._batchnorm = batchnorm
        self._bn_layer = None

    def __call__(self, inputs):
        if isinstance(inputs, list):
            assert len(inputs) == 1, len(inputs)
            inputs = inputs[0]
        out = inputs  # output from previous layer
        if self._batchnorm:
            do_scope = self._bn_layer is None
            if do_scope:
                self._bn_layer = BatchNormalization()
            out = self._bn_layer([out])
            if do_scope and self.scope:
                self._bn_layer._bn._name = (self.scope + '/' +
                                            self._bn_layer._bn._name)
                if self._bn_layer._activation:
                    self._bn_layer._activation._name = (
                            self.scope + '/' + self._bn_layer._activation._name)
        if self._dropout:
            do_scope = self._dropout_layer is None
            if do_scope:
                self._dropout_layer = Dropout(rate=self._dropout)
            out = self._dropout_layer([out])
            if do_scope and self.scope:
                self._dropout_layer._layer._name = (
                        self.scope + '/' + self._dropout_layer._layer._name)
        return out


def _dynamic_block_factory(cls):
    class _SomeBlock(cls):
        def __init__(self, *args, **kwargs):
            self.scope = kwargs.pop('scope', None)
            dropout = kwargs.pop('dropout', None)
            batchnorm = kwargs.pop('batchnorm', True)
            self._nullable = kwargs.pop('nullable', False)
            self._activation_first = kwargs.pop('activation_first', False)
            self._activation_layer = None
            if self._activation_first:
                self._activation = kwargs.pop('activation')
            else:
                self._activation = kwargs.pop('activation', None)
            self._add_layer = None
            self._block_layer = BlockOps(dropout=dropout, batchnorm=batchnorm,
                                         scope=self.scope)
            super().__init__(*args, **kwargs)

        def __call__(self, inputs, *args, **kwargs):
            if self._nullable and len(inputs) == 0:
                return []
            if len(inputs) > 1:
                do_scope = self._add_layer is None
                if do_scope:
                    self._add_layer = AddStrictly()
                inputs = [self._add_layer(inputs)]
                if do_scope and self.scope:
                    self._add_layer._layer._name = (
                            self.scope + '/' + self._add_layer._layer._name)
                    if self._add_layer._activation_layer:
                        self._add_layer._activation_layer._name = (
                                self.scope + '/' +
                                self._add_layer._activation_layer._name)
            if self._activation_first:
                do_scope = self._activation_layer is None
                if do_scope:
                    self._activation_layer = tf.keras.layers.Activation(
                        self._activation)
                inputs = [self._activation_layer(inputs[0])]
                if do_scope and self.scope:
                    self._activation_layer._name = (
                            self.scope + '/' + self._activation_layer._name)
            do_scope = hasattr(self, '_layer') and self._layer is None
            out = super().__call__(inputs, *args, **kwargs)
            if do_scope and self.scope:
                self._layer._name = self.scope + '/' + self._layer._name
            out = self._block_layer(out)
            if not self._activation_first and self._activation is not None:
                do_scope = self._activation_layer is None
                if do_scope:
                    self._activation_layer = tf.keras.layers.Activation(
                        self._activation)
                out = self._activation_layer(out)
                if do_scope and self.scope:
                    self._activation_layer._name = (
                            self.scope + '/' + self._activation_layer._name)
            return out

    _SomeBlock.__name__ = cls.__name__ + 'Block'

    return _SomeBlock


Conv2DBlock = _dynamic_block_factory(Conv2D)
SeparableConv2DBlock = _dynamic_block_factory(SeparableConv2D)
DenseBlock = _dynamic_block_factory(Dense)
AvgPool2DBlock = _dynamic_block_factory(AvgPool2D)
IdentityBlock = _dynamic_block_factory(Identity)


class NullableIdentity(Identity):
    def __call__(self, inputs, **kwargs):
        if len(inputs) == 0:
            return []
        else:
            return super().__call__(inputs, **kwargs)


class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        self._axis = axis
        super().__init__(**kwargs)

    def _compute_transpose_axes(self, input_shape):
        rank = len(input_shape)
        axis = self._axis
        if axis < 0:
            axis = rank + axis
            assert axis >= 0
        else:
            assert axis < rank
        if axis < rank - 1:
            # we need to transpose axes (swap axes)
            axes = [*range(rank)]
            axes[axis] = rank - 1
            axes[-1] = axis
        else:
            axes = None
        return axes

    def compute_output_shape(self, input_shape):
        axes = self._compute_transpose_axes(input_shape)
        if axes:
            input_shape = [input_shape[i] for i in axes]
        return input_shape

    def call(self, inputs):  # noqa
        axes = self._compute_transpose_axes(inputs.get_shape())
        if axes:
            return tf.transpose(inputs, axes)
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config['axis'] = self._axis
        return config


class DenseOnAxis:
    def __init__(self, *args, **kwargs):
        self._axis = kwargs.pop('axis', -1)
        self._dense_args = args
        self._dense_kwargs = kwargs

    def __call__(self, inputs):
        inputs = TransposeLayer(axis=self._axis)(inputs)

        # Dense on last axis
        outputs = tf.keras.layers.Dense(
            *self._dense_args, **self._dense_kwargs
        )(inputs)

        outputs = TransposeLayer(axis=self._axis)(outputs)
        return outputs


class AddByProjectingND(AddByProjecting):
    """"""

    def __call__(self, values, seed=None, **kwargs):
        # case where there is no inputs
        if len(values) == 0:
            return []

        values = values[:]
        max_len_shp = max(len(x.get_shape()) for x in values)

        # projection
        if len(values) > 1:

            for i, v in enumerate(values):

                if len(v.get_shape()) < max_len_shp:
                    values[i] = tf.keras.layers.Reshape(
                        (
                            *tuple(v.get_shape()[1:]),
                            *tuple(1 for _ in
                                   range(max_len_shp - len(v.get_shape()))),
                        )
                    )(v)

            shapes = np.asarray([[*v.get_shape()[1:]] for v in values])
            proj_shape = scipy.stats.mode(shapes, axis=0)[0].squeeze(axis=0)

            for i, v in enumerate(values):
                shape = v.get_shape()[1:]
                for k, (d_v, d_p) in enumerate(zip(shape, proj_shape)):
                    if d_v != d_p:  # ensure each dim has correct shape
                        v = DenseOnAxis(
                            units=d_p,
                            axis=k + 1,
                            kernel_initializer=tf.keras.initializers.glorot_uniform(
                                seed=seed
                            ),
                        )(v)
                        values[i] = v

        # concatenation
        if len(values) > 1:
            out = tf.keras.layers.Add()(values)
            if self.activation is not None:
                out = tf.keras.layers.Activation(self.activation)(out)
        else:
            out = values[0]
        return out


class GlobalAveragePooling2D(Operation):
    def __init__(  # noqa
            self,
            data_format=None,
            keepdims=False,
            nullable=False,
            **kwargs,
    ):
        self.data_format = data_format
        self.keepdims = keepdims
        self.nullable = nullable
        if keepdims:
            logger.warning('Keepdims not supported')
        self.kwargs = kwargs
        self._layer = None

    def __str__(self):
        return f"GlobalAveragePooling2D"

    def __call__(self, inputs, **kwargs):
        """Create the tensorflow operation.

        Args:
            inputs (list(Tensor)): list of input tensors.

        Return: a tensor corresponding to the operation.
        """
        if self.nullable:
            if len(inputs) == 0:
                return []
        assert (
                len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:
            self._layer = tf.keras.layers.GlobalAveragePooling2D(
                data_format=self.data_format,
                **self.kwargs,
            )
        out = self._layer(inputs[0])
        return out


class AddStrictly(Operation):
    def __init__(  # noqa
            self,
            activation=None,
            scope=None,
            **kwargs,
    ):
        self.kwargs = kwargs
        self.scope = scope
        self._layer = None
        self._activation = activation
        self._activation_layer = None

    def __str__(self):
        return f"AddStrictly"

    def __call__(self, inputs, **kwargs):
        """Create the tensorflow operation.

        Args:
            inputs (list(Tensor)): list of input tensors.

        Return: a tensor corresponding to the operation.
        """
        if len(inputs) == 0:
            return []
        elif len(inputs) == 1:
            return inputs[0]
        if self._layer is None:
            self._layer = tf.keras.layers.Add(
                **self.kwargs,
            )
        out = self._layer(inputs)
        if self._activation is not None:
            do_scope = self._activation_layer is None
            if do_scope:
                self._activation_layer = tf.keras.layers.Activation(
                    self._activation)
            out = self._activation_layer(out)
            if do_scope and self.scope:
                self._activation_layer._name = (
                        self.scope + '/' + self._activation_layer._name)
        return out


def add_dense_to_(node: VariableNode):
    activations = [tf.nn.relu, tf.nn.tanh]
    dropout_rates = [None, 0.5, 0.8]
    for units in [1024, 512, 128]:
        for activation in activations:
            for rate in dropout_rates:
                # node.add_op(MaybeFlattenThenDense(
                #     units=units, activation=activation))
                node.add_op(DenseBlock(units=units,
                                       activation=activation,
                                       dropout=rate))


def add_conv2d_to_(node: VariableNode):
    activations = [tf.nn.relu, tf.nn.tanh]
    kernel_sizes = [3, 5]
    filters_szs = [64, 128, 256]
    dropout_rates = [None, 0.5, 0.8]

    for kernel_size in kernel_sizes:
        for filters in filters_szs:
            for activation in activations:
                for rate in dropout_rates:
                    node.add_op(Conv2DBlock(kernel_size=kernel_size,
                                            filters=filters,
                                            activation=activation,
                                            dropout=rate))


def add_pool2d_to_(node: VariableNode):
    pool_sizes = [2, 4]
    pool_classes = [partial(AvgPool2D, padding='valid'), MaxPool2D]

    for pool_size in pool_sizes:
        for pool_class in pool_classes:
            node.add_op(pool_class(pool_size=pool_size))


class NASBlock(ABC):
    nodes = None

    def __init__(self):
        self._build()
        assert self.nodes is not None, 'Bad subclass, bad!'

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    @property
    def input_node(self):
        return self.nodes[0]

    @property
    def output_node(self):
        return self.nodes[-1]


class NASBench201Cell(NASBlock):
    def __init__(
            self,
            arch,
            filters,
            n_nodes=4,  # V
            cell_edges=None,
            scope=None,
    ):
        self.arch = arch
        self.filters = filters
        self.n_nodes = n_nodes
        self.scope = scope
        if cell_edges is None:
            cell_edges = {}
            for i in range(1, n_nodes):
                for j in range(0, i):
                    # shh i know it's a node just shh
                    cell_edges[(i, j)] = VariableNode()
        self.cell_edges = cell_edges
        self.nodes = [ConstantNode(AddStrictly()) for _ in range(n_nodes)]
        super().__init__()

    def _build(self):
        arch = self.arch
        filters = self.filters
        n_nodes = self.n_nodes
        nodes = self.nodes
        cell_edges = self.cell_edges

        ops = [
            lambda: Conv2DBlock(  # 3x3 conv2d block
                kernel_size=3,
                filters=filters,
                padding='SAME',
                activation='relu',
                activation_first=True,
                batchnorm=True,
                dropout=None,
                nullable=True,
                scope=self.scope,
            ),
            lambda: Conv2DBlock(  # 1x1 conv2d block
                kernel_size=1,
                filters=filters,
                padding='SAME',
                activation='relu',
                activation_first=True,
                batchnorm=True,
                dropout=None,
                nullable=True,
                scope=self.scope,
            ),
            lambda: AvgPool2DBlock(  # 3x3 avgpool2d block
                pool_size=3,
                padding='SAME',
                batchnorm=False,
                dropout=None,
                nullable=True,
                scope=self.scope,
            ),
            lambda: Tensor([]),  # zeroize
            lambda: NullableIdentity(),  # skip connection
        ]
        for i in range(1, n_nodes):
            node_i = nodes[i]

            for j in range(0, i):
                node_j = nodes[j]
                cell_edge = cell_edges[(i, j)]
                for op in ops:
                    cell_edge.add_op(op())
                arch.connect(node_j, cell_edge)
                arch.connect(cell_edge, node_i)

    def mimed_copy(self, filters=None, scope=None):
        cell_edges = {
            k: MimeNode(v) for k, v in self.cell_edges.items()
        }
        return self.__class__(
            arch=self.arch,
            filters=filters or self.filters,
            n_nodes=self.n_nodes,
            cell_edges=cell_edges,
            scope=scope or self.scope,
        )


class NASBench201ResidualBlock(NASBlock):
    """
    The intermediate residual block is the basic residual block with a stride
    of 2 (He et al., 2016), which serves to downsample the spatial size and
    double the channels of an input feature map. The shortcut path in this
    residual block consists of a 2-by-2 average pooling layer with stride of 2
    and a 1-by-1 convolution.
    """

    def __init__(self, arch, filters, strides=2, scope=None):
        self.arch = arch
        self.filters = filters
        self.strides = strides
        self.scope = scope
        super().__init__()

    def _build(self):
        # main path
        main_conv1 = ConstantNode(Conv2DBlock(
            kernel_size=3,
            filters=self.filters,
            strides=self.strides,
            activation='relu',
            padding='SAME',
            batchnorm=True,
            dropout=None,
            nullable=True,
            scope=self.scope,
        ))
        main_conv2 = ConstantNode(Conv2DBlock(
            kernel_size=3,
            filters=self.filters,
            activation=None,
            padding='SAME',
            batchnorm=True,
            dropout=None,
            nullable=True,
            scope=self.scope,
        ))
        self.arch.connect(main_conv1, main_conv2)
        # shortcut path
        shortcut_pool = ConstantNode(AvgPool2DBlock(
            pool_size=2,
            strides=self.strides,
            batchnorm=False,
            dropout=None,
            nullable=True,
            scope=self.scope,
        ))
        shortcut_conv = ConstantNode(Conv2DBlock(
            kernel_size=1,
            filters=self.filters,
            activation=None,
            padding='SAME',
            batchnorm=False,
            dropout=None,
            nullable=True,
            scope=self.scope,
        ))
        self.arch.connect(shortcut_pool, shortcut_conv)
        # putting it together
        input_node = ConstantNode(NullableIdentity())
        self.arch.connect(input_node, main_conv1)
        self.arch.connect(input_node, shortcut_pool)
        merge = ConstantNode(AddStrictly(activation='relu', scope=self.scope))
        self.arch.connect(main_conv2, merge)
        self.arch.connect(shortcut_conv, merge)
        # the nodes of the network
        self.nodes = [input_node, main_conv1, main_conv2, shortcut_conv,
                      shortcut_pool, merge]


def nas_bench_201_search_space(
        input_shape,
        output_shape,
        regression=False,
        n_nodes=4,
        stack_size=5,
        filter_scale_factor=1,
):
    """
    The NAS-Bench-201 search space
    https://arxiv.org/pdf/2001.00326.pdf

    Args:
        input_shape: the input shape
        output_shape: the output shape
        regression: whether the task is regression
        n_nodes: the number of nodes in a cell (V)
        stack_size: the number of cells in a stack/block (N)
        filter_scale_factor: the scale factor for the number of filters in
            each conv block, e.g., for full training of a candidate good arch

    Returns:
        DeepHyper compatible search space (AutoKSearchSpace)
    """
    arch = AutoKSearchSpace(input_shape, output_shape, regression=regression)
    source = arch.input_nodes[0]

    # Macro skeleton
    # first conv
    channels_stage_1 = round(16 * filter_scale_factor)
    conv1 = ConstantNode(Conv2DBlock(
        kernel_size=3,
        filters=channels_stage_1,
        activation=None,
        dropout=None,
        batchnorm=True,
        scope='initial_conv'
    ))
    arch.connect(source, conv1)
    prev_node = conv1
    # cell stack 1
    core_cell = None
    scope = 'nasbench201_block_1'
    for _ in range(stack_size):
        if core_cell is None:
            core_cell = cell = NASBench201Cell(arch, filters=channels_stage_1,
                                               n_nodes=n_nodes, scope=scope)
        else:
            cell = core_cell.mimed_copy(filters=channels_stage_1, scope=scope)
        arch.connect(prev_node, cell.input_node)
        prev_node = cell.output_node
    # residual block 1
    channels_stage_2 = round(32 * filter_scale_factor)
    residual1 = NASBench201ResidualBlock(
        arch=arch, filters=channels_stage_2, strides=2, scope='residual_1'
    )
    arch.connect(prev_node, residual1.input_node)
    prev_node = residual1.output_node
    # cell stack 2
    scope = 'nasbench201_block_2'
    for _ in range(stack_size):
        cell = core_cell.mimed_copy(filters=channels_stage_2, scope=scope)
        arch.connect(prev_node, cell.input_node)
        prev_node = cell.output_node
    # residual block 2
    channels_stage_3 = round(64 * filter_scale_factor)
    residual2 = NASBench201ResidualBlock(
        arch=arch, filters=channels_stage_3, strides=2, scope='residual_2'
    )
    arch.connect(prev_node, residual2.input_node)
    prev_node = residual2.output_node
    # cell stack 3
    scope = 'nasbench201_block_3'
    for _ in range(stack_size):
        cell = core_cell.mimed_copy(filters=channels_stage_3, scope=scope)
        arch.connect(prev_node, cell.input_node)
        prev_node = cell.output_node
    # global average pooling
    gap = ConstantNode(GlobalAveragePooling2D(nullable=True))
    arch.connect(prev_node, gap)

    return arch
