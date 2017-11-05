from __future__ import print_function
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.utils import shuffle

class INFOGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.num_classes = 10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 140

        optimizer = Adam(0.0001, 0.5)
        losses = ['binary_crossentropy', 'categorical_crossentropy', self.gaussian_loss]

        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        # self.discriminator.compile(loss=losses,
        #                            optimizer=optimizer,
        #                            metrics=['accuracy'])
        # check if the pre-trained weights exist to lead
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
        if os.path.exists("./saved_model/discriminator.h5"):
            print("loading weights for discriminator")
            self.discriminator.load_weights('./saved_model/discriminator.h5')
        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'],optimizer=optimizer)
        if os.path.exists("./saved_model/generator.h5"):
            print("loading weights for generator")
            self.generator.load_weights('./saved_model/generator.h5')
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label, target_cont = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(gen_input, [valid, target_label, target_cont])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)
        if os.path.exists("./saved_model/adversarial.h5"):
            print("loading weights for the adversarial")
            self.combined.load_weights('./saved_model/adversarial.h5')

    # Reference: https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/InfoGAN/
    def gaussian_loss(self, y_true, y_pred):

        mean = y_pred[0]
        log_stddev = y_pred[1]
        y_true = y_true[0]

        epsilon = (y_true - mean) / (K.exp(log_stddev) + K.epsilon())
        loss = (log_stddev + 0.5 * K.square(epsilon))

        return K.mean(loss)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(1024, activation='relu', input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256 * 4 * 4, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((4, 4, 256)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=4, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        gen_input = Input(shape=(self.latent_dim,))
        img = model(gen_input)

        return Model(gen_input, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.summary()

        img = Input(shape=self.img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)

        def linmax(x):
            return K.maximum(x, -16)

        def linmax_shape(input_shape):
            return input_shape

        c_model = Dense(128)(features)
        c_model = LeakyReLU(alpha=0.2)(c_model)
        c_model = BatchNormalization(momentum=0.8)(c_model)

        label = Dense(self.num_classes, activation="softmax")(c_model)

        mean = Dense(1, activation="linear")(c_model)
        log_stddev = Dense(1)(c_model)
        log_stddev = Lambda(linmax, output_shape=linmax_shape)(log_stddev)

        cont = concatenate([mean, log_stddev], axis=1)

        return Model(img, [validity, label, cont])

    def sample_generator_input(self, batch_size,y_train_batch):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 128))
        # sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        # sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)
        true_labels = to_categorical(y_train_batch, num_classes=self.num_classes)
        sampled_cont = np.random.uniform(-1, 1, size=(batch_size, 2))
        return sampled_noise, true_labels, sampled_cont

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train,y_train = shuffle(X_train,y_train,random_state=0)
        X_test,y_test = shuffle(X_test,y_test,random_state=0)

        # Rescale -1 to 1
        X_train = X_train.astype(np.float32)/255
        X_test = X_test.astype(np.float32)/255
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        X_valid = X_test[-1000:]
        y_valid = y_test[-1000:]

        half_batch = int(batch_size / 2)
        nb = int(X_train.shape[0]/batch_size)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for b in range(nb):
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                # Train discriminator on generator output
                sampled_noise, sampled_labels, sampled_cont = self.sample_generator_input(half_batch,y_train[idx])
                gen_input = np.concatenate((sampled_noise, sampled_labels, sampled_cont), axis=1)
                # Generate a half batch of new images
                gen_imgs = self.generator.predict(gen_input)
                fake = np.zeros((half_batch, 1))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels, sampled_cont])

                # Train discriminator on real data
                # Select a random half batch of images
                imgs = X_train[idx]
                labels = to_categorical(y_train[idx], num_classes=self.num_classes)
                valid = np.ones((half_batch, 1))
                d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels, sampled_cont])

                # Avg. loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                valid = np.ones((half_batch, 1))
                sampled_noise, sampled_labels, sampled_cont = self.sample_generator_input(half_batch,y_train[idx])
                gen_input = np.concatenate((sampled_noise, sampled_labels, sampled_cont), axis=1)

                # Train the generator
                g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels, sampled_cont])

                # Plot the progress
                if (b % (nb/2) == 0):
                    _,p_Y_valid,_ = self.discriminator.predict_on_batch(X_valid)
                    acc = utils.accuracy(y_valid,p_Y_valid)
                    print("Epoch: %d [D loss: %.2f, acc.: %.2f%%, label_acc: %.2f%%] [G loss: %.2f]" % (
                    epoch, d_loss[0], 100 * d_loss[4], 100 * d_loss[5], g_loss[0]))
                    print("Testing Accuracy: {} %".format(acc))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch,X_train,y_train)
            if epoch % 1000 == 0:
                self.save_model()

    def save_imgs(self, epoch,X,y):
        r, c = 10, 10
        idx = np.random.randint(0, X.shape[0], c)
        fig, axs = plt.subplots(r, c)
        for i in range(r):
            sampled_noise, sampled_labels, sampled_cont = self.sample_generator_input(c,y[idx])
            gen_input = np.concatenate((sampled_noise, sampled_labels, sampled_cont), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5
            for j in range(c):
                axs[i, j].imshow(gen_imgs[j, :, :, :])
                axs[i, j].axis('off')
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            weights_path = "./saved_model/%s.h5" % model_name
            model.save_weights(weights_path)
        print("Saving weights.......")
        save(self.generator, "generator")
        save(self.discriminator, "discriminator")
        save(self.combined, "adversarial")


if __name__ == '__main__':
    utils.setup_logging()
    infogan = INFOGAN()
    infogan.train(epochs=200, batch_size=64, save_interval=10)