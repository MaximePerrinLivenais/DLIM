from tensorflow import keras
from tensorflow.keras import layers

from downsampling_block import DownsamplingBlock

class Discriminator(keras.Model):
    def __init__(self, input_size, filters = 64, nb_downsampling_blocks = 3,
                    kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02),
                    name = None):
        super(Discriminator, self).__init__()
        input_tensor = layers.Input(shape = input_size, name = name + '_img_input')
        input_conv_tensor = layers.Conv2D(filters, (4, 4), strides = (2, 2), padding = 'same',
                                          kernel_initializer = kernel_initializer)(input_tensor)
        tensor = layers.LeakyReLU(0.2)(input_conv_tensor)
        downsampling_filters = filters
        for downsampling_block_index in range(nb_downsampling_blocks):
            downsampling_filters *= 2
            strides = (2, 2) if downsampling_block_index < 2 else (1, 1)
            tensor = DownsamplingBlock(downsampling_filters, layers.LeakyReLU(0.2),
                                        kernel_size = (4, 4), strides = strides)(tensor)

        output_conv_tensor = layers.Conv2D(1, (4, 4), strides = (1, 1), padding = 'same',
                                            kernel_initializer = kernel_initializer)(tensor)
        self.model = keras.models.Model(inputs = input_tensor, outputs = output_conv_tensor, name = name)

    def call(self, input, training = False):
        return self.model(input, training)
