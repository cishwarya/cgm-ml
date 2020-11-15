import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from PIL import Image

        
class VariationalAutoencoder(tf.keras.Model):

    def __init__(self, input_shape, filters, latent_dim, size):
        super().__init__()

        self.filters = []
        self.latent_dim = latent_dim
        assert size in ["tiny", "small", "big"]
        self.size = size

        # TODO Describe.
        bridge_shape = (input_shape[0] // 2**len(filters), input_shape[1] // 2**len(filters), filters[-1])

        if self.size == "tiny":
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

            self.decoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(bridge_shape), activation="relu"),
                tf.keras.layers.Reshape(bridge_shape),
                tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(2, 2), padding="same", activation="linear")
            ])
        elif self.size == "small":
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

            self.decoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(bridge_shape), activation="relu"),
                tf.keras.layers.Reshape(bridge_shape),
                tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(2, 2), padding="same", activation="linear")
            ])
        elif self.size == "big":
            self.encoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[4], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[5], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters=filters[6], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

            self.decoder = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=np.prod(bridge_shape), activation="relu"),
                tf.keras.layers.Reshape(bridge_shape),
                tf.keras.layers.Conv2DTranspose(filters=filters[5], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[4], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[3], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=3, strides=(2, 2), padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=(2, 2), padding="same", activation="linear")
            ])
        else:
            assert False, self.size


    def call(self, x):

        # Encode. Compute mean and variance.
        mean, logvar = self.encode(x)

        # Get latent vector.
        z = self.reparameterize(mean, logvar)

        # Decode
        y = self.decode(z, apply_sigmoid=True)

        return y

    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    
    def train(self, dataset_train, dataset_validate, epochs, batch_size, shuffle_buffer_size, render=False):
        optimizer = tf.keras.optimizers.Adam(1e-4)
        print("Starting training...")

        dataset_train_samples = None
        for x in dataset_train.batch(10).take(1):
            dataset_train_samples = x[0:10]
            break
        
        dataset_validate_samples = None
        for x in dataset_validate.batch(10).take(1):
            dataset_validate_samples = x[0:10]
            break
        #dataset_samples = list(dataset_samples.as_numpy_iterator())
        #dataset_samples = np.array(dataset_samples)
        
        dataset_train = dataset_train.shuffle(shuffle_buffer_size).batch(batch_size)
        dataset_validate = dataset_validate.shuffle(shuffle_buffer_size).batch(batch_size)

        if render:
            render_reconstructions(self, dataset_train_samples, filename=f"reconstruction-train-0000.png")
            render_reconstructions(self, dataset_validate_samples, filename=f"reconstruction-validate-0000.png")

        for epoch in range(1, epochs + 1):

            start_time = time.time()
            for train_x in dataset_train:
                train_step(self, train_x, optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for validate_x in dataset_validate:
                loss(compute_loss(self, validate_x))
            elbo = -loss.result()
            #display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                    .format(epoch, elbo, end_time - start_time))

            if render:
                render_reconstructions(self, dataset_train_samples, filename=f"reconstruction-train-{epoch:04d}.png")
                render_reconstructions(self, dataset_validate_samples, filename=f"reconstruction-validate-{epoch:04d}.png")
        
        if render:
            create_animation("reconstruction-train-*", "reconstruction-animation-train.gif")
            create_animation("reconstruction-validate-*", "reconstruction-animation-validate.gif")


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def render_reconstructions(model, samples, filename):

    reconstructions = model.predict(samples, steps=10)
    
    image = np.zeros((2 * samples.shape[1], samples.shape[0]  * samples.shape[1], 3))
    for sample_index, (sample, reconstruction) in enumerate(zip(samples, reconstructions)):
        s1 = 0
        e1 = sample.shape[1]
        s2 = sample_index * sample.shape[0]
        e2 = (sample_index + 1) * sample.shape[0]
        image[s1:e1, s2:e2] = sample
        s1 = sample.shape[1]
        e1 = 2 * sample.shape[1]
        s2 = sample_index * sample.shape[0]
        e2 = (sample_index + 1) * sample.shape[0]
        image[s1:e1, s2:e2] = reconstruction
    
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(filename)


def create_animation(glob_search_path, filename):
    with imageio.get_writer(filename, mode="I") as writer:
        paths = glob.glob(glob_search_path)
        paths = sorted(paths)
        for path in paths:
            image = imageio.imread(path)
            writer.append_data(image)
        image = imageio.imread(path)
        writer.append_data(image)


