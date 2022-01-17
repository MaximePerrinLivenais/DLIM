class ResnetGenerator(keras.Model):
    def __init__(self, filters = 64, nb_downsampling_blocks = 2, nb_residual_blocks = 9,
                  nb_upsampling_blocks = 2, gamma_initializer = gamma_init, name = None):
        super(ResnetGenerator, self).__init__()

        input_layer = layers.Input(shape = input_img_size, name = name + "_img_input")
        input_reflection_layer = ReflectionPadding2D(padding = (3, 3))(input_layer)
        input_conv_layer = layers.Conv2D(filters, (7, 7), kernel_initializer = kernel_init, use_bias = False)(input_reflection_layer)
        norm_layer = tfa.layers.InstanceNormalization(gamma_initializer = kernel_init)(input_conv_layer)
        previous_layer = layers.Activation('relu')(norm_layer)

        for _ in range(nb_downsampling_blocks):
            filters *= 2
            previous_layer = downsample(previous_layer, filters = filters, activation = layers.Activation('relu'))

        for _ in range(nb_residual_blocks):
            previous_layer = residual_block(previous_layer, activation = layers.Activation('relu'))

        for _ in range(nb_upsampling_blocks):
            filters //= 2
            previous_layer = upsample(previous_layer, filters, activation = layers.Activation('relu'))

        output_reflection_layer = ReflectionPadding2D(padding = (3, 3))(previous_layer)
        output_conv_layer = layers.Conv2D(3, (7, 7), padding = 'valid')(output_reflection_layer)
        activation_layer = layers.Activation('tanh')(output_conv_layer)

        self.model = keras.models.Model(input_layer, activation_layer, name = name)

    def call(self, input, training = False):
        return self.model(input, training)
