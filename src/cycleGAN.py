import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import Progbar

class CycleGAN(keras.Model):
    def __init__(self, target_style_generator, source_style_generator,
                  target_style_discriminator, source_style_discriminator,
                  lambda_cycle = 10.0, lambda_identity = 0.5):
        super(CycleGAN, self).__init__()
        self.target_style_gen = target_style_generator
        self.src_style_gen = source_style_generator
        self.target_style_disc = target_style_discriminator
        self.src_style_disc = source_style_discriminator
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, target_style_gen_optimizer, src_style_gen_optimizer,
                  target_style_disc_optimizer, src_style_disc_optimizer,
                  generator_loss_function, discriminator_loss_function):
        super(CycleGAN, self).compile()
        self.target_style_gen_opt = target_style_gen_optimizer
        self.src_style_gen_opt = src_style_gen_optimizer
        self.target_style_disc_opt = target_style_disc_optimizer
        self.src_style_disc_opt = src_style_disc_optimizer
        self.gen_loss_function = generator_loss_function
        self.disc_loss_function = discriminator_loss_function
        self.cycle_loss_function = keras.losses.MeanAbsoluteError()
        self.identity_loss_function = keras.losses.MeanAbsoluteError()

    def predict(self, input):
        return self.target_style_gen.predict(input)

    def cycle(self, input):
        return self.src_style_gen(self.target_style_gen(input))

    def transform(self, input_batch, target_style_generator, source_style_generator,
                    target_style_discriminator, source_style_discriminator, training = False):
        target_style_batch = target_style_generator(input_batch, training)
        cycled_batch = source_style_generator(target_style_batch, training)
        identity_batch = source_style_generator(input_batch, training)

        input_discrimination = source_style_discriminator(input_batch, training)
        target_style_discrimination = target_style_discriminator(target_style_batch, training)

        return target_style_batch, cycled_batch, identity_batch, target_style_discrimination, input_discrimination

    def compute_generator_loss(self, input, identity, cycle, discrimination):
        adverserial_loss = self.gen_loss_function(discrimination)
        cycle_loss = self.cycle_loss_function(input, cycle) * self.lambda_cycle
        identity_loss = self.identity_loss_function(input, identity) * self.lambda_cycle * self.lambda_identity

        return adverserial_loss + cycle_loss + identity_loss

    def compute_gradient(self, model, loss, optimizer, gradient_tape):
        gradient = gradient_tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    def train_step(self, data_batch):
        # X and Y are respectively source and target style.
        X_input, Y_input = data_batch

        with tf.GradientTape(persistent=True) as tape:
            X_transformation = self.transform(X_input, self.target_style_gen, self.src_style_gen,
                                                self.target_style_disc, self.src_style_disc, True)
            Y_fake, X_cycle, X_identity, Y_fake_disc, X_input_disc = X_transformation
            Y_transformation = self.transform(Y_input, self.src_style_gen, self.target_style_gen,
                                                self.src_style_disc, self.target_style_disc, True)
            X_fake, Y_cycle, Y_identity, X_fake_disc, Y_input_disc = Y_transformation

            target_style_gen_loss = self.compute_generator_loss(Y_input, Y_identity, Y_cycle, Y_fake_disc)
            src_style_gen_loss = self.compute_generator_loss(X_input, X_identity, X_cycle, X_fake_disc)
            target_style_disc_loss = self.disc_loss_function(Y_input_disc, Y_fake_disc)
            src_style_disc_loss = self.disc_loss_function(X_input_disc, X_fake_disc)

        #self.compute_gradient(self.gen_G, total_loss_G, self.gen_G_optimizer, tape)
        #self.compute_gradient(self.gen_F, total_loss_F, self.gen_F_optimizer, tape)
        #self.compute_gradient(self.disc_X, disc_X_loss, self.disc_X_optimizer, tape)
        #self.compute_gradient(self.disc_Y, disc_Y_loss, self.disc_Y_optimizer, tape)

        target_style_gen_grads = tape.gradient(target_style_gen_loss, self.target_style_gen.trainable_variables)
        src_style_gen_grads = tape.gradient(src_style_gen_loss, self.src_style_gen.trainable_variables)
        target_style_disc_grads = tape.gradient(target_style_disc_loss, self.target_style_disc.trainable_variables)
        src_style_disc_grads = tape.gradient(src_style_disc_loss, self.src_style_disc.trainable_variables)

        self.target_style_gen_opt.apply_gradients(zip(target_style_gen_grads, self.target_style_gen.trainable_variables))
        self.src_style_gen_opt.apply_gradients(zip(src_style_gen_grads, self.src_style_gen.trainable_variables))
        self.target_style_disc_opt.apply_gradients(zip(target_style_disc_grads, self.target_style_disc.trainable_variables))
        self.src_style_disc_opt.apply_gradients(zip(src_style_disc_grads, self.src_style_disc.trainable_variables))

        return [target_style_gen_loss, src_style_gen_loss, target_style_disc_loss, src_style_disc_loss]

    def compute_val_loss(self, data_batch):
        # X and Y are respectively source and target style.
        X_input, Y_input = data_batch
        _, _, _, Y_fake_disc, _ = self.transform(X_input, self.target_style_gen, self.src_style_gen,
                                                  self.target_style_disc, self.src_style_disc)
        _, Y_cycle, Y_identity, _, _ = self.transform(Y_input, self.src_style_gen, self.target_style_gen,
                                                        self.src_style_disc, self.target_style_disc)
        return self.compute_generator_loss(Y_input, Y_identity, Y_cycle, Y_fake_disc)

    def fit_test(self, input_data, target_data, nb_epochs, nb_batch_per_epoch,
                  input_val_data = None, target_val_data = None, nb_batch_per_val = 0,
                  save_weight_filepath = None):
        data = zip(input_data, target_data)
        metrics_name = ['target generator', 'source generator', 'target discriminator', 'source discriminator']

        if input_val_data and target_val_data and nb_batch_per_val > 0:
            validation_data = zip(input_val_data, target_val_data)
            metrics_name.append('validation loss')
            if save_weight_filepath:
                best_val_loss = None
        elif input_val_data or target_val_data or nb_batch_per_val > 0 or save_weight_filepath:
            raise Exception('Missing arguments to have proper validation mecanism.')
        else:
            validation_data = None

        for epoch in range(nb_epochs):
            print('\nepoch {}/{}'.format(epoch + 1, nb_epochs))
            progress_bar = Progbar(nb_batch_per_epoch, stateful_metrics = metrics_name)
            epoch_metrics = [tf.keras.metrics.Mean() for _ in range(4)]

            for batch_index, batch in enumerate(data):
                if batch_index >= nb_batch_per_epoch:
                    break

                batch_losses_list = self.train_step(batch)
                for index, batch_loss in enumerate(batch_losses_list):
                    epoch_metrics[index](batch_loss)

                epoch_losses = [metric.result().numpy().mean() for metric in epoch_metrics]
                print(type(epoch_losses))
                progress_bar.update(batch_index, values = zip(metrics_name, epoch_losses), finalize = validation_data is None)

            if validation_data is None:
                continue

            validation_metric = tf.keras.metrics.Mean()
            for batch_index, batch in enumerate(validation_data):
                if batch_index >= nb_batch_per_val:
                    break
                validation_metric(self.compute_val_loss(batch))

            epoch_losses = epoch_losses.append(validation_metric.result().numpy().mean()))
            print(type(epoch_losses))
            progress_bar.update(nb_batch_per_epoch, values = zip(metrics_name, epoch_losses), finalize = True)

            if save_weight_filepath and (best_val_loss is None or val_loss <= best_val_loss):
                self.save_weights(save_weight_filepath)
