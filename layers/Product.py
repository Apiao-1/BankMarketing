import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal, glorot_uniform)


class InnerProductLayer(Layer):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.

      Input shape
        - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape: ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.

      Arguments
        - **reduce_sum**: bool. Whether return inner product or element-wise product

      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, reduce_sum=True, **kwargs):
        self.reduce_sum = reduce_sum
        super(InnerProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `InnerProductLayer` layer should be called '
                             'on a list of at least 2 inputs')

        reduced_inputs_shapes = [shape.as_list() for shape in input_shape]
        shape_set = set()

        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_inputs_shapes[i]))

        if len(shape_set) > 1:
            raise ValueError('A `InnerProductLayer` layer requires '
                             'inputs with same shapes '
                             'Got different shapes: %s' % (shape_set))

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A `InnerProductLayer` layer requires '
                             'inputs of a list with same shape tensor like (None,1,embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))
        super(InnerProductLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([embed_list[idx] for idx in row], axis=1)  # batch num_pairs k
        q = tf.concat([embed_list[idx] for idx in col], axis=1)

        inner_product = p * q
        if self.reduce_sum:
            inner_product = tf.reduce_sum(inner_product, axis=2, keep_dims=True)
        return inner_product

    def compute_output_shape(self, input_shape):
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        input_shape = input_shape[0]
        embed_size = input_shape[-1]
        if self.reduce_sum:
            return (input_shape[0], num_pairs, 1)
        else:
            return (input_shape[0], num_pairs, embed_size)

    def get_config(self, ):
        config = {'reduce_sum': self.reduce_sum, }
        base_config = super(InnerProductLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

