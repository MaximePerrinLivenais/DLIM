from tensorflow import keras
from tensorflow.keras import layers

from downsampling_block import DownsamplingBlock
from reflection_padding_layer import ReflectionPadding2D
from residual_block import ResidualBlock
from upsampling_block import UpsamplingBlock

import tensorflow_addons as tfa

class ResnetGenerator(keras.Model):
    def __init__(self, input_size, filters = 64, nb_downsampling_blocks = 2, nb_residual_blocks = 9, nb_upsampling_blocks = 2,
                 kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02),
                 gamma_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02), name = None):
        super(ResnetGenerator, self).__init__()

        input_tensor = layers.Input(shape = input_size, name = name + "_img_input")
        input_reflection_tensor = ReflectionPadding2D(padding = (3, 3))(input_tensor)
        input_conv_tensor = layers.Conv2D(filters, (7, 7), kernel_initializer = kernel_initializer, use_bias = False)(input_reflection_tensor)
        norm_tensor = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)(input_conv_tensor)
        tensor = layers.Activation('relu')(norm_tensor)

        for _ in range(nb_downsampling_blocks):
            filters *= 2
            tensor = DownsamplingBlock(filters, activation = layers.Activation('relu'))(tensor)

        for _ in range(nb_residual_blocks):
            tensor = ResidualBlock(256, activation = layers.Activation('relu'))(tensor)

        for _ in range(nb_upsampling_blocks):
            filters //= 2
            tensor = UpsamplingBlock(filters, activation = layers.Activation('relu'))(tensor)

        output_reflection_tensor = ReflectionPadding2D(padding = (3, 3))(tensor)
        output_conv_tensor = layers.Conv2D(3, (7, 7), padding = 'valid')(output_reflection_tensor)
        activation_tensor = layers.Activation('tanh')(output_conv_tensor)

        self.model = keras.models.Model(input_tensor, activation_tensor, name = name)

    def call(self, input_tensor, training = False):
        return self.model(input_tensor, training)
