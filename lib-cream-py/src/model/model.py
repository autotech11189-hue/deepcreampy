import os

import tensorflow as tf
from keras import Model, Input
from keras.src import optimizers, layers
from keras.src.saving import load_model

from .contextual_block import ContextualBlock
from .decoder import Decoder
from .disciminator_red import DiscriminatorRed
from .encoder import Encoder

class InpaintNN:
    def __init__(self, model_path: str, input_height=256, input_width=256, batch_size=1, create_model=False):
        super(InpaintNN, self).__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.model_path = model_path
        self.create_model = create_model
        if not create_model:
            self.check_model_file()
            self.model = load_model(self.model_path)

    def check_model_file(self):
        if not os.path.exists(self.model_path):
            print("\nMissing Model, download model")
            print("Read: https://github.com/deeppomf/DeepCreamPy/blob/master/docs/INSTALLATION.md#run-code-yourself \n")
            exit(-1)

    def train(self, epochs: int, dataset):
        #todo: make class based & have different response shapes for training & execution
        X = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        Y = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        MASK = Input(shape=(self.input_height, self.input_width, 3), batch_size=self.batch_size, dtype=tf.float32)
        # todo: unclear
        IT = 0

        # Model layers
        input_tensor = layers.Concatenate(axis=-1)([X, MASK])
        encoder = Encoder(name='G_en')
        vec_en = encoder(input_tensor)
        cb1 = ContextualBlock(name='CB1', k_size=3, lamda=50.0, stride=1)
        vec_con = cb1((vec_en, vec_en, MASK))

        decoder = Decoder(name='G_de', size1=self.input_height, size2=self.input_height)
        I_co = decoder(vec_en)
        I_ge = decoder(vec_con)

        image_result = I_ge * (1 - MASK) + Y * MASK

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

        model = Model(inputs=[X, Y, MASK], outputs=[image_result, I_ge, I_co])
        checkpoint_path = os.path.join(self.model_path, "model_checkpoint")
        checkpoint = tf.train.Checkpoint(generator=model, discriminator=disc_red, optimizer_G=optimizer_G,
                                         optimizer_D=optimizer_D)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Checkpoint restored from {checkpoint_manager.latest_checkpoint}")

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
            print(f"\nStart epoch {epoch}")

            for step, (real_images, y, masks) in enumerate(dataset):
                d_loss, g_loss = train_step(real_images, y, masks)

                if step % 100 == 0:
                    print(
                        f"Step {step}: Discriminator Loss: {d_loss.numpy():.4f}, Generator Loss: {g_loss.numpy():.4f}")

            checkpoint_manager.save()
            print(f"Checkpoint saved at {checkpoint_manager.latest_checkpoint}")

    def load_checkpoint(self):
        checkpoint_path = tf.train.latest_checkpoint(self.model_path)

        if checkpoint_path:
            self.model.load_weights(checkpoint_path)
            print(f"Model restored from {checkpoint_path}")
        else:
            print("\nNo checkpoint found")
            exit(-1)

    def predict_image(self, censored, unused, mask):
        image_result, _, _ = self.model(censored, unused, mask, training=False)
        return image_result


if __name__ == "__main__":
    m = InpaintNN("", create_model=True)
    m.model.summary(expand_nested=True)
