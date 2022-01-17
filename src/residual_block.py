from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
from reflection_padding_layer import ReflectionPadding2D

class ResidualBlock(layers.Layer):
    def __init__(self, dimension, activation, kernel_size = (3, 3), strides = (1, 1), padding = 'valid',
                  kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02),
                  gamma_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02),
                  use_bias = False):
        super(ResidualBlock, self).__init__()
        self.input_reflection_layer = ReflectionPadding2D()
        self.input_conv_layer = layers.Conv2D(dimension, kernel_size, strides = strides,
                                              kernel_initializer = kernel_initializer, padding = padding,
                                              use_bias = use_bias)
        self.input_norm_layer = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)
        self.activation_layer = activation
        self.output_reflection_layer = ReflectionPadding2D()
        self.output_conv_layer = layers.Conv2D(dimension, kernel_size, strides = strides,
                                                kernel_initializer = kernel_initializer, padding = padding,
                                                use_bias = use_bias)
        self.output_norm_layer = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)

    def call(self, input_tensor):
        tensor = self.input_reflection_layer(input_tensor)
        tensor = self.input_conv_layer(tensor)
        tensor = self.input_norm_layer(tensor)
        tensor = self.activation_layer(tensor)
        tensor = self.output_reflection_layer(tensor)
        tensor = self.output_conv_layer(tensor)
        tensor = self.output_norm_layer(tensor)
        return layers.add([input_tensor, tensor])
