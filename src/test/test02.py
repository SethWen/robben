"""  
    author: Shawn
    time  : 11/9/18 7:18 PM
    desc  :
    update: Shawn 11/9/18 7:18 PM      
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils, layers, Input, callbacks


def construct_net():
    inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor
    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return inputs, outputs


def train():
    inputs, outputs = construct_net()
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains for 5 epochs
    # model.fit(data, labels, batch_size=32, epochs=5)


def show_network(model):
    utils.plot_model(model, to_file='modelme.png', show_shapes=True)
    pass


class MyModel(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # Define your layers here.
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  callbacks.TensorBoard(log_dir='./logs')
]
# model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
#           validation_data=(val_data, val_labels))


# show_network(model)

if __name__ == '__main__':
    pass
