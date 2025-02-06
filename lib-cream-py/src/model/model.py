import os

import tensorflow as tf
from keras import Model, Input
from keras.src import optimizers, layers
from keras.src.saving import load_model

from logger import Logger
from .contextual_block import ContextualBlock
from .decoder import Decoder
from .disciminator_red import DiscriminatorRed
from .encoder import Encoder

class InpaintModel(Model):
    def __init__(self, input_height=256, input_width=256, **kwargs):
        super(InpaintModel, self).__init__(name="", **kwargs)
        self.input_height = input_height
        self.input_width = input_width
        self.encoder = Encoder(name='G_en')
        self.cb1 = ContextualBlock(name='CB1', k_size=3, lamda=50.0, stride=1)
        self.decoder = Decoder(name='G_de', size1=self.input_height, size2=self.input_height)

    def call(self, inputs, training=False):
        X, Y, MASK = inputs
        input_tensor = layers.Concatenate(axis=-1)([X, MASK])
        vec_en = self.encoder(input_tensor)
        vec_con = self.cb1((vec_en, vec_en, MASK))

        I_co = self.decoder(vec_en)
        I_ge = self.decoder(vec_con)

        image_result = I_ge * (1 - MASK) + Y * MASK
        if training:
            return image_result, I_ge, I_co
        else:
            return image_result


class InpaintNN:
    def __init__(self, model_path: str, input_height=256, input_width=256, batch_size=1, create_model=False, logger=Logger()):
        super(InpaintNN, self).__init__()

        self.new_name_map = None
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.create_model = create_model
        self.logger = logger
        if not create_model:
            self.check_model_file()
            self.model = load_model(self.model_path)

    def check_model_file(self):
        if not os.path.exists(self.model_path):
            self.logger.error("\nMissing Model, download model")
            self.logger.error("Read: https://github.com/deeppomf/DeepCreamPy/blob/master/docs/INSTALLATION.md#run-code-yourself \n")
            exit(-1)

    def migrate_weights(self):
        model_path = '../models/mosaic/Train_290000'
        #model_path =  '../models/bar/Train_775000'
        reader = tf.compat.v1.train.NewCheckpointReader(model_path)
        variable_map = reader.get_variable_to_shape_map()
        lookup_model = {
            'G_en/conv2d/kernel': 'G_en/Conv/weights',
            'G_en/conv2d/bias': 'G_en/Conv/biases',
            'G_en/conv2d_1/kernel': 'G_en/Conv_1/weights',
            'G_en/conv2d_1/bias': 'G_en/Conv_1/biases',
            'G_en/conv2d_2/kernel': 'G_en/Conv_2/weights',
            'G_en/conv2d_2/bias': 'G_en/Conv_2/biases',
            'G_en/conv2d_3/kernel': 'G_en/Conv_3/weights',
            'G_en/conv2d_3/bias': 'G_en/Conv_3/biases',
            'G_en/conv2d_4/kernel': 'G_en/Conv_4/weights',
            'G_en/conv2d_4/bias': 'G_en/Conv_4/biases',
            'G_en/conv2d_5/kernel': 'G_en/Conv_5/weights',
            'G_en/conv2d_5/bias': 'G_en/Conv_5/biases',
            'G_en/conv2d_6/kernel': 'G_en/Conv_6/weights',
            'G_en/conv2d_6/bias': 'G_en/Conv_6/biases',
            'G_en/conv2d_7/kernel': 'G_en/Conv_7/weights',
            'G_en/conv2d_7/bias': 'G_en/Conv_7/biases',
            'G_en/conv2d_8/kernel': 'G_en/Conv_8/weights',
            'G_en/conv2d_8/bias': 'G_en/Conv_8/biases',
            'G_en/conv2d_9/kernel': 'G_en/Conv_9/weights',
            'G_en/conv2d_9/bias': 'G_en/Conv_9/biases',
            'G_de/conv_nn/conv2d_10/kernel': 'G_de/Conv/weights',
            'G_de/conv_nn/conv2d_10/bias': 'G_de/Conv/biases',
            'G_de/conv_nn/conv2d_11/kernel': 'G_de/Conv_1/weights',
            'G_de/conv_nn/conv2d_11/bias': 'G_de/Conv_1/biases',
            'G_de/conv_nn_1/conv2d_12/kernel': 'G_de/Conv_2/weights',
            'G_de/conv_nn_1/conv2d_12/bias': 'G_de/Conv_2/biases',
            'G_de/conv_nn_1/conv2d_13/kernel': 'G_de/Conv_3/weights',
            'G_de/conv_nn_1/conv2d_13/bias': 'G_de/Conv_3/biases',
            'G_de/conv_nn_2/conv2d_14/kernel': 'G_de/Conv_4/weights',
            'G_de/conv_nn_2/conv2d_14/bias': 'G_de/Conv_4/biases',
            'G_de/conv_nn_2/conv2d_15/kernel': 'G_de/Conv_5/weights',
            'G_de/conv_nn_2/conv2d_15/bias': 'G_de/Conv_5/biases',
            'G_de/conv_nn_3/conv2d_16/kernel': 'G_de/Conv_6/weights',
            'G_de/conv_nn_3/conv2d_16/bias': 'G_de/Conv_6/biases',
            'G_de/conv_nn_3/conv2d_17/kernel': 'G_de/Conv_7/weights',
            'G_de/conv_nn_3/conv2d_17/bias': 'G_de/Conv_7/biases',
            'G_de/conv2d_18/kernel': 'G_de/Conv_8/weights',
            'G_de/conv2d_18/bias': 'G_de/Conv_8/biases',
            'CB1/ML/kernel': 'CB1/ML/weights',
            'CB1/ML/bias': 'CB1/ML/biases'
        }
        lookup_disc = {
            'disc_red/l1/w': 'disc_red/l1w',
            'disc_red/l1/b': 'disc_red/l1b',
            'disc_red/l1/wu': 'disc_red/l1wu',
            'disc_red/l2/w': 'disc_red/l2w',
            'disc_red/l2/b': 'disc_red/l2b',
            'disc_red/l2/wu': 'disc_red/l2wu',
            'disc_red/l3/w': 'disc_red/l3w',
            'disc_red/l3/b': 'disc_red/l3b',
            'disc_red/l3/wu': 'disc_red/l3wu',
            'disc_red/l4/w': 'disc_red/l4w',
            'disc_red/l4/b': 'disc_red/l4b',
            'disc_red/l4/wu': 'disc_red/l4wu',
            'disc_red/l5/w': 'disc_red/l5w',
            'disc_red/l5/b': 'disc_red/l5b',
            'disc_red/l5/wu': 'disc_red/l5wu',
            'disc_red/l6/w': 'disc_red/l6w',
            'disc_red/l6/b': 'disc_red/l6b',
            'disc_red/l6/wu': 'disc_red/l6wu',
            'disc_red/l7/_w': 'disc_red/l7_w',
            'disc_red/l7/_b': 'disc_red/l7_b',
            'disc_red/l7/w_0u': 'disc_red/l7w_0u',
            'disc_red/l7/w_1u': 'disc_red/l7w_1u',
            'disc_red/l7/w_2u': 'disc_red/l7w_2u',
            'disc_red/l7/w_3u': 'disc_red/l7w_3u',
            'disc_red/l7/w_4u': 'disc_red/l7w_4u',
            'disc_red/l7/w_5u': 'disc_red/l7w_5u',
            'disc_red/l7/w_6u': 'disc_red/l7w_6u',
            'disc_red/l7/w_7u': 'disc_red/l7w_7u',
            'disc_red/l7/w_8u': 'disc_red/l7w_8u',
            'disc_red/l7/w_9u': 'disc_red/l7w_9u',
            'disc_red/l7/w_10u': 'disc_red/l7w_10u',
            'disc_red/l7/w_11u': 'disc_red/l7w_11u',
            'disc_red/l7/w_12u': 'disc_red/l7w_12u',
            'disc_red/l7/w_13u': 'disc_red/l7w_13u',
            'disc_red/l7/w_14u': 'disc_red/l7w_14u',
            'disc_red/l7/w_15u': 'disc_red/l7w_15u'
        }
        variable_map = dict([(name, variable_map[name]) for name in variable_map if not name.__contains__("Adam")])

        for var in self.model.variables:
            old_name = lookup_model[var.path]
            old_shape = variable_map.pop(old_name)
            assert old_shape == var.shape
            v = reader.get_tensor(old_name)
            var.assign(v)
        for var in self.disc_red.variables:
            old_name = lookup_disc[var.path]
            old_shape = variable_map.pop(old_name)
            assert old_shape == var.shape
            v = reader.get_tensor(old_name)
            var.assign(v)
        variable_map.pop('beta1_power')
        variable_map.pop('beta2_power')
        variable_map.pop('beta1_power_1')
        variable_map.pop('beta2_power_1')
        if len(variable_map) != 0:
            import pprint
            pprint.pprint(variable_map)
            raise Exception("Variable map is not empty")
        self.model.save(self.model_path)

    def train(self, epochs: int, dataset, checkpoint_path: str):
        X = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        Y = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        MASK = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        # todo: unclear
        IT = 0

        output = InpaintModel(self.input_height, self.input_width)((X, Y, MASK))

        # Discriminator
        disc_red = DiscriminatorRed(name='disc_red')

        # todo never used?
        # SSIM (Structural Similarity Index)
        # A = tf.image.rgb_to_yuv((image_result + 1) / 2.0)
        # A_Y = tf.cast(A[:, :, :, 0:1] * 255.0, tf.int32)
        # B = tf.image.rgb_to_yuv((X + 1) / 2.0)
        # B_Y = tf.cast(B[:, :, :, 0:1] * 255.0, tf.int32)
        # ssim = tf.reduce_mean(tf.image.ssim(tf.cast(A_Y, tf.float32), tf.cast(B_Y, tf.float32), max_val=255.0))

        alpha = IT / 1000000

        # Optimizers
        optimizer_D = optimizers.Adam(learning_rate=0.0004, beta_1=0.5,
                                      beta_2=0.9)  # .minimize(Loss_D, var_list=var_D)
        optimizer_G = optimizers.Adam(learning_rate=0.0001, beta_1=0.5,
                                      beta_2=0.9)  # .minimize(Loss_G, var_list=var_G)

        model = Model(inputs=[X, Y, MASK], outputs=[output])

        checkpoint = tf.train.Checkpoint(generator=model, discriminator=disc_red, optimizer_G=optimizer_G,
                                         optimizer_D=optimizer_D)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            self.logger.debug(f"Checkpoint restored from {checkpoint_manager.latest_checkpoint}")

        @tf.function
        def train_step(real_images, y, masks):
            with tf.GradientTape() as tape_D, tf.GradientTape() as tape_G:
                fake_images, I_ge, I_co = model([real_images, y, masks], training=True)

                D_real = disc_red(real_images)
                D_fake = disc_red(fake_images)

                loss_D = tf.reduce_mean(tf.nn.relu(1 + D_fake)) + tf.reduce_mean(tf.nn.relu(1 - D_real))
                loss_GAN = -tf.reduce_mean(D_fake)
                loss_s_re = tf.reduce_mean(tf.abs(I_ge - real_images))
                loss_hat = tf.reduce_mean(tf.abs(I_co - real_images))

                loss_G = 0.1 * loss_GAN + 10 * loss_s_re + 5 * (1 - alpha) * loss_hat

            # Compute gradients
            grads_D = tape_D.gradient(loss_D, disc_red.trainable_variables)
            grads_G = tape_G.gradient(loss_G, model.trainable_variables)

            # Apply gradients
            optimizer_D.apply_gradients(zip(grads_D, disc_red.trainable_variables))
            optimizer_G.apply_gradients(zip(grads_G, model.trainable_variables))

            return loss_D, loss_G

        # Training loop
        for epoch in range(epochs):
            self.logger.info(f"\nStart epoch {epoch}")

            for step, (real_images, y, masks) in enumerate(dataset):
                d_loss, g_loss = train_step(real_images, y, masks)

                if step % 100 == 0:
                    self.logger.info(
                        f"Step {step}: Discriminator Loss: {d_loss.numpy():.4f}, Generator Loss: {g_loss.numpy():.4f}")

            checkpoint_manager.save()
            self.logger.info(f"Checkpoint saved at {checkpoint_manager.latest_checkpoint}")

        if epochs == 0:
            optimizer_G.build(model.trainable_variables)
            disc_red.build((None, 256, 256, 3))
            optimizer_D.build(disc_red.trainable_variables)
        else:
            model.save(self.model_path)
        optimizer_G_vars = [x.path for x in optimizer_G.variables]
        optimizer_D_vars = [x.path for x in optimizer_D.variables]
        new_name_map = {}
        for i, var_name in enumerate(optimizer_G_vars):
            checkpoint_name = f"(root).optimizer_G._variables.{i}"
            new_name_map[checkpoint_name] = var_name

        for i, var_name in enumerate(optimizer_D_vars):
            checkpoint_name = f"(root).optimizer_D._variables.{i}"
            new_name_map[checkpoint_name] = var_name
        self.new_name_map = new_name_map
        self.model = model
        self.disc_red = disc_red

    def load_checkpoint(self):
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)

        if checkpoint_path:
            self.model.load_weights(checkpoint_path)
            self.logger.info(f"Model restored from {checkpoint_path}")
        else:
            self.logger.warn("\nNo checkpoint found")
            exit(-1)

    def predict_image(self, censored, unused, mask):
        image_result = self.model((censored, unused, mask), training=False)
        return image_result


if __name__ == "__main__":
    m = InpaintNN("", create_model=True)
    m.model.summary(expand_nested=True)
