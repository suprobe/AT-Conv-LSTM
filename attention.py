from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==2

        self.W_0 = self.add_weight(name='att_weight0',
                                 shape=(input_shape[0][1], input_shape[0][1]),
                                 initializer='uniform',
                                 trainable=True)
        self.W_1 = self.add_weight(name='att_weight1',
                                 shape=(input_shape[1][1], input_shape[1][1]),
                                 initializer='uniform',
                                 trainable=True)

        self.W_2 = self.add_weight(name='att_weight2',
                                   shape=(input_shape[0][1], input_shape[0][1]),
                                   initializer='uniform',
                                   trainable=True)
        # self.b = self.add_weight(name='att_bias',
        #                          shape=(input_shape[0][1],),
        #                          initializer='uniform',
        #                          trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x1 = K.permute_dimensions(inputs[0], (0, 1))
        x2 = K.permute_dimensions(inputs[1][:,-1,:], (0, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x1, self.W_0)+ K.dot(x2, self.W_1)))
        a = K.dot(a, self.W_2)
        outputs = K.permute_dimensions(a * x1, (0, 1))
        # outputs = K.sum(outputs, axis=1)
        outputs = K.l2_normalize(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]