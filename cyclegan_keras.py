
"""
This script implements the CycleGAN model for a dataset of MR and CT .nii images. This script 
is based off an implementation for low-resolution 2D images by the Tensorflow authors,
found here: https://www.tensorflow.org/tutorials/generative/cyclegan 
The links for the original CycleGAN proposal are below:
- [Paper](https://arxiv.org/pdf/1703.10593.pdf)
- [Original implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
"""

"""
## Setup
"""

import os
import numpy as np
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import NiftiGenerator
import nibabel as nib
from itertools import chain

autotune = tf.data.experimental.AUTOTUNE
tf.config.run_functions_eagerly(True)


"""
Prepare the dataset
"""


# Size of the random crops to be used during training.
input_img_size = (512,512,3) # orign. (256,256,3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# lists for plotting generator & discriminator losses
plot_g, plot_f, plot_x, plot_y, plotlist = [],[],[],[],[]
# lists to collect losses over each batch
ls_g, ls_f, ls_x, ls_y = [], [], [], []
# set to use SingleNiftiGen (unpaired data) or PairedNiftiGen (paired data)
SingleNiftiGen = False

"""
SingleNiftiGenerator (Unpaired Data)
"""

if SingleNiftiGen:

    # Set up NiftiGenerators
    X_gen, Y_gen = NiftiGenerator.SingleNiftiGenerator(), NiftiGenerator.SingleNiftiGenerator()

    # Augmentation
    X_gen_aug_opts = NiftiGenerator.SingleNiftiGenerator.get_default_augOptions()
    Y_gen_aug_opts = NiftiGenerator.SingleNiftiGenerator.get_default_augOptions()
    #X_gen_aug_opts.addnoise, Y_gen_aug_opts.addnoise = 1e-9, 1e-9
    X_gen_aug_opts.hflips, Y_gen_aug_opts.hflips = True, True
    X_gen_aug_opts.vflips, Y_gen_aug_opts.vflips = True, True
    X_gen_aug_opts.rotations, Y_gen_aug_opts.rotations = 15, 15
    X_gen_aug_opts.scalings, Y_gen_aug_opts.scalings = 0.2, 0.2
    X_gen_aug_opts.shears, Y_gen_aug_opts.shears = 2, 2
    X_gen_aug_opts.translations, Y_gen_aug_opts.translations = 32, 32

    # Normalization
    X_gen_norm_ops = NiftiGenerator.SingleNiftiGenerator.get_default_normOptions()
    Y_gen_norm_ops = NiftiGenerator.SingleNiftiGenerator.get_default_normOptions()
    X_gen_norm_ops.normXtype = 'auto'
    Y_gen_norm_ops.normXtype = 'fixed'
    Y_gen_norm_ops.normXoffset = 1000
    Y_gen_norm_ops.normXscale = 2500

    # Second normalization method
    #Y_gen_norm_ops.normXtype = 'none'
    # def correct_CT_vals(CT_img):
    #   new_img = CT_img
    #   new_img += 1000
    #   new_img /= 2500
    #   new_img[CT_img <= 0] = 0
    #   return new_img
    # Y_gen_aug_opts.additionalFunction = correct_CT_vals # Unpaired data

    # Initialization and generation
    X_gen.initialize('/nii/t1_bravo_mr_pair.txt', X_gen_aug_opts, X_gen_norm_ops) 
    Y_gen.initialize('/nii/t1_bravo_ct_pair.txt', Y_gen_aug_opts, Y_gen_norm_ops)

    X_iterator = X_gen.generate_chunks(chunk_size=(64,64,3), batch_size=1) 
    Y_iterator = Y_gen.generate_chunks(chunk_size=(64,64,3), batch_size=1)

    # Wrap the two NiftGens into one to pass to model
    iterator = NiftiGenerator.UnpairedDualNiftiGenerator.generate(X_iterator,Y_iterator)


"""
PairedNiftiGenerator (Paired/Registered Data)
"""

if not SingleNiftiGen:
    
    # Set up NiftiGenerator
    niftiGen = NiftiGenerator.PairedNiftiGenerator()

    # Augmentation -- paired data
    niftiGen_aug_opts = NiftiGenerator.PairedNiftiGenerator.get_default_augOptions()
    niftiGen_aug_opts.hflips = True
    niftiGen_aug_opts.vflips = True
    niftiGen_aug_opts.rotations = 15
    niftiGen_aug_opts.scalings = 0.2
    niftiGen_aug_opts.shears = 2
    niftiGen_aug_opts.translations = 32

    # Normalization -- paired data
    niftiGen_norm_opts = NiftiGenerator.PairedNiftiGenerator.get_default_normOptions()
    niftiGen_norm_opts.normXtype = 'auto'
    niftiGen_norm_opts.normYtype = 'fixed'
    niftiGen_norm_opts.normYoffset = 1000
    niftiGen_norm_opts.normYscale = 2500

    # Initialization and generation - using voxel normalized images
    niftiGen.initialize(
        '/nii/t1_bravo_mr_voxelnorm.txt', 
        '/nii/t1_bravo_ct_voxelnorm.txt', 
        augOptions=niftiGen_aug_opts, 
        normOptions=niftiGen_norm_opts,
        batchTransformFunction=None)
    # The iterator object is passed to the model
    iterator = niftiGen.generate_chunks(chunk_size=(64,64,3), batch_size=1) 


"""
## Building blocks used in the CycleGAN generators and discriminators
"""

class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


"""
## Build the generators

The generator consists of downsampling blocks: nine residual blocks
and upsampling blocks. The structure of the generator is the following:

```
c7s1-64 ==> Conv block with `relu` activation, filter size of 7
d128 ====|
         |-> 2 downsampling blocks
d256 ====|
R256 ====|
R256     |
R256     |
R256     |
R256     |-> 9 residual blocks
R256     |
R256     |
R256     |
R256 ====|
u128 ====|
         |-> 2 upsampling blocks
u64  ====|
c7s1-3 => Last conv block with `tanh` activation, filter size of 7.
```
"""


def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


"""
## Build the discriminators

The discriminators implement the following architecture:
`C64->C128->C256->C512`
"""


def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")


"""
## Build the CycleGAN model

We will override the `train_step()` method of the `Model` class
for training via `fit()`.
"""


class CycleGan(keras.Model):

    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    # To prevent error of not having 'call' method when subclassing 'Model' class
    def call(self, inputs):
        return None

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            
            fake_y = self.gen_G(real_x, training=True)
            fake_x = self.gen_F(real_y, training=True)

            # Cycle x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        # Define losses as class attributes to access for loss plots
        self.total_loss_G = total_loss_G
        self.total_loss_F = total_loss_F
        self.disc_X_loss = disc_X_loss
        self.disc_Y_loss = disc_Y_loss

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }



"""
## Create a callback that periodically saves generated images
"""

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_batch_end(self, batch, logs=None):

        # Collect losses for each batch for plotting
        ls_g.append(float(self.model.total_loss_G.numpy()))
        ls_f.append(float(self.model.total_loss_F.numpy()))
        ls_x.append(float(self.model.disc_X_loss.numpy()))
        ls_y.append(float(self.model.disc_Y_loss.numpy()))

    def on_epoch_end(self, epoch, logs=None):

        # Plot in-progress image
        img = next(iterator) # shape (2,1,512,512,3)
        prediction = self.model.gen_G(img)[0].numpy() # shape (512,512,3)
        input_nii = nib.Nifti1Image(np.asarray(img), np.eye(4))
        pred_nii = nib.Nifti1Image(prediction, np.eye(4))
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).astype(np.uint8)

        plt.figure(1, figsize=(12, 12))
        display_list = [img[0,:,:,0], prediction]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i])# * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig('/nii/Results/INP_img_epoch_' + str(epoch+1) + '.png', bbox_inches = 'tight', pad_inches = 0.1)

        nib.save(input_nii, os.path.join('/nii/Results', 'INP_input_epoch_' + str(epoch+1)))
        nib.save(pred_nii, os.path.join('/nii/Results', 'INP_pred_epoch_' + str(epoch+1)))

        # Average batch losses over whole epoch
        plot_g.append(np.mean(ls_g))
        plot_f.append(np.mean(ls_f))
        plot_x.append(np.mean(ls_x))
        plot_y.append(np.mean(ls_y))
        plotlist.append(epoch + 1)

        # Progress of generator losses
        plt.clf()
        fig2 = plt.figure(2, figsize=(16,8))
        plt.plot(plotlist, plot_g)
        plt.plot(plotlist, plot_f)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['Gen_G: MR->CT', 'Gen_F: CT->MR'])
        plt.title("Generator losses")
        fig2.savefig('/nii/Generator losses.png', bbox_inches = 'tight', pad_inches = 0.1)

        # Progress of discriminator losses
        plt.clf()
        fig3 = plt.figure(3, figsize=(16,8))
        plt.plot(plotlist, plot_x, 'g')
        plt.plot(plotlist, plot_y, 'r')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['Disc_MR', 'Disc_CT'])
        plt.title('Discriminator losses')
        fig3.savefig('/nii/Discriminator losses.png', bbox_inches = 'tight', pad_inches = 0.1)

        # Clear batch loss lists for next epoch
        ls_g.clear()
        ls_f.clear()
        ls_x.clear()
        ls_y.clear()


"""
## Train the end-to-end model
"""


# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)

# Load checkpoint weights if training again
# weight_file = "/nii/Results/cyclegan_checkpoints.033"
# cycle_gan_model.load_weights(weight_file).expect_partial()
# print("Weights loaded successfully")

# Callbacks
plotter = GANMonitor()
checkpoint_filepath = '/nii/Results/cyclegan_checkpoints.{epoch:03d}'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath
)

cycle_gan_model.fit(
    iterator, 
    epochs=200,
    steps_per_epoch=256, # no. of slices/image
    callbacks=[model_checkpoint_callback, plotter]
)
