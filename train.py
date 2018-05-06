from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


DC_D_W1 = tf.Variable(xavier_init([5, 5, 1, 16]))
DC_D_b1 = tf.Variable(tf.zeros(shape=[16]))

DC_D_W2 = tf.Variable(xavier_init([3, 3, 16, 32]))
DC_D_b2 = tf.Variable(tf.zeros(shape=[32]))

DC_D_W3 = tf.Variable(xavier_init([7 * 7 * 32, 128]))
DC_D_b3 = tf.Variable(tf.zeros(shape=[128]))

DC_D_W4 = tf.Variable(xavier_init([128, 1]))
DC_D_b4 = tf.Variable(tf.zeros(shape=[1]))

theta_DC_D = [DC_D_W1, DC_D_b1, DC_D_W2, DC_D_b2, DC_D_W3, DC_D_b3, DC_D_W4, DC_D_b4]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def dc_discriminator(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(tf.nn.conv2d(x, DC_D_W1, strides=[1, 2, 2, 1], padding='SAME') + DC_D_b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, DC_D_W2, strides=[1, 2, 2, 1], padding='SAME') + DC_D_b2)
    conv2 = tf.reshape(conv2, shape=[-1, 7 * 7 * 32])
    h = tf.nn.relu(tf.matmul(conv2, DC_D_W3) + DC_D_b3)
    logit = tf.matmul(h, DC_D_W4) + DC_D_b4
    prob = tf.nn.sigmoid(logit)

    return prob, logit



def generator(network, nb_classes, input_shape, z):
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    #model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    noise = Input(z)
    G_prob = model(noise)

    return model


def train_and_score(network, dataset):

    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()

    model = compile_model(network, nb_classes, input_shape)
    G_sample = generator(network, nb_classes, input_shape, Z)
    D_real, D_logit_real = dc_discriminator(X)
    D_fake, D_logit_fake = dc_discriminator(G_sample)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
