"""  
    author: Shawn
    time  : 10/1/18 7:19 PM
    desc  :
    update: Shawn 10/1/18 7:19 PM      
"""

import random

import numpy as np
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from IPython.display import Image
import tensorflow as tf
from tensorflow.keras import utils, models, layers

from src.constant import *


def gen(batch_size=32):
    X = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, CHARACTERS_LENGTH), dtype=np.uint8) for i in range(CAPTCHA_LENGTH)]
    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(CHARACTERS) for j in range(4)])
            print(random_str)
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, CHARACTERS.find(ch)] = 1
        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([CHARACTERS[x] for x in y])


# X, y = next(gen(2))
# plt.imshow(X[0])
# plt.title(decode(y))
# plt.show()
# plt.close()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    # keras.layers.Convolution2D(32 * 2 ** 1, 3, 3, activation=tf.nn.relu),
    # keras.layers.Convolution2D(32 * 2 ** 1, 3, 3, activation=tf.nn.relu),
    # keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(CHARACTERS_LENGTH, activation=tf.nn.softmax)
])

input_tensor = layers.Input((60, 170, 3))
x = input_tensor
# x = layers.Convolution2D(32 * 2 ** 1, 3, 3, activation='relu')(x)
# x = layers.Convolution2D(32 * 2 ** 2, 3, 3, activation='relu')(x)

for i in range(2):
    x = layers.Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
    x = layers.Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
    # x = layers.MaxPooling2D((2, 2))(x)

# x = layers.Flatten()(x)
# x = layers.Dropout(0.25)(x)
#
# x = [layers.Dense(CAPTCHA_LENGTH, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]

# model = models.Model(input=input_tensor, output=x)

# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

# model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
#                     nb_worker=2, pickle_safe=True,
#                     validation_data=gen(), nb_val_samples=1280)


def show_network():
    utils.plot_model(model, to_file='model.png', show_shapes=True)
    pass


if __name__ == '__main__':
    # show_network()
    # X, y = next(gen(1))
    # plt.imshow(X[0])
    # plt.title(decode(y))
    # plt.show()
    pass
