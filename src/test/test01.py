"""  
    author: Shawn
    time  : 11/9/18 6:41 PM
    desc  :
    update: Shawn 11/9/18 6:41 PM      
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils, layers

print(tf.VERSION)
print(tf.keras.__version__)


def show_network(model):
    utils.plot_model(model, to_file='model.png', show_shapes=True)
    pass


def construct_net():
    # method 1
    # model = tf.keras.Sequential()
    # # Adds a densely-connected layer with 64 units to the model:
    # model.add(layers.Dense(64, activation='relu'))
    # # Add another:
    # model.add(layers.Dense(64, activation='relu'))
    # # Add a softmax layer with 10 output units:
    # model.add(layers.Dense(10, activation='softmax'))

    # method 2
    model = tf.keras.Sequential([
        # Adds a densely-connected layer with 64 units to the model:
        layers.Dense(64, activation='relu'),
        # Add another:
        layers.Dense(64, activation='relu'),
        # Add a softmax layer with 10 output units:
        layers.Dense(10, activation='softmax'),
        layers.Dense(10, activation='softmax'),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_layer():
    # Create a sigmoid layer:
    layers.Dense(64, activation='sigmoid')
    # Or:
    layers.Dense(64, activation=tf.sigmoid)

    # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
    layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

    # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
    layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

    # A linear layer with a kernel initialized to a random orthogonal matrix:
    layers.Dense(64, kernel_initializer='orthogonal')

    # A linear layer with a bias vector initialized to 2.0s:
    layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

    print('---------------------')


def generate_data():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    # data = np.random.randn(100, 32).astype(np.float32)
    # labels = np.random.randn(100, 10).astype(np.float32)
    return data, labels


def train1():
    model = construct_net()
    data, labels = generate_data()
    val_data, val_labels = generate_data()
    model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))


def train2():
    model = construct_net()
    data, labels = generate_data()
    val_data, val_labels = generate_data()

    # tf.cast(data, tf.float64)

    # Instantiates a toy dataset instance:
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()

    # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
    model.fit(dataset, epochs=10, steps_per_epoch=30)


def evaluate():
    # model.evaluate(data, labels, batch_size=32)
    # model.evaluate(dataset, steps=30)
    pass


def predict():
    # result = model.predict(data, batch_size=32)
    # print(result.shape)
    pass


if __name__ == '__main__':
    # show_network(construct_net())
    # create_layer()
    # train1()
    train2()
    pass
