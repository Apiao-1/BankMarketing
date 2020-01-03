import tensorflow as tf


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, **kwargs):

        self.l2_reg = l2_reg
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")
        self.mode = mode
        self.use_bias = use_bias
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        if self.mode != 0:
            self.dense = tf.keras.layers.Dense(units=1, activation=None, use_bias=False,
                                               kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if self.mode == 0:
            sparse_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            linear_logit = self.dense(dense_input)
        else:
            sparse_input, dense_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=False) + self.dense(dense_input)
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
