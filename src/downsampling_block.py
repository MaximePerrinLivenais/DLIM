from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

class DownsamplingBlock(layers.Layer):
    def __init__(self, filters, activation, kernel_size = (3, 3), strides = (2, 2), padding = 'same',
                 kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02),
                 gamma_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02),
                 use_bias = False):
        super(DownsamplingBlock, self).__init__()
        self.conv_layer = layers.Conv2D(filters, kernel_size, strides = strides,
                                          kernel_initializer = kernel_initializer, padding = padding,
                                          use_bias = use_bias)
        self.norm_layer = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)
        self.activation_layer = activation

    def call(self, input_tensor):
        tensor = self.conv_layer(input_tensor)
        tensor = self.norm_layer(tensor)
        return self.activation_layer(tensor) if self.activation_layer else tensor
