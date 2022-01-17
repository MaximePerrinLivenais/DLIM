class Discriminator(keras.Model):
    def __init__(self, filters = 64, kernel_initializer = kernel_init,
                    nb_downsampling_blocks = 3, name = None):
        super(Discriminator, self).__init__()
        input_layer = layers.Input(shape = input_img_size, name = name + '_img_input')
        input_conv_layer = layers.Conv2D(filters, (4, 4), strides = (2, 2), padding = 'same',
                                          kernel_initializer = kernel_initializer)(input_layer)
        previous_layer = layers.LeakyReLU(0.2)(input_conv_layer)
        downsampling_filters = filters
        for downsampling_block_index in range(nb_downsampling_blocks):
            downsampling_filters *= 2
            strides = (2, 2) if downsampling_block_index < 2 else (1, 1)
            previous_layer = downsample(previous_layer, filters = downsampling_filters,
                                        activation = layers.LeakyReLU(0.2),
                                        kernel_size = (4, 4), strides = strides)

        output_conv_layer = layers.Conv2D(1, (4, 4), strides = (1, 1), padding = 'same',
                                            kernel_initializer = kernel_initializer)(previous_layer)
        self.model = keras.models.Model(inputs = input_layer, outputs = output_conv_layer, name = name)

    def call(self, input, training = False):
        return self.model(input, training)
