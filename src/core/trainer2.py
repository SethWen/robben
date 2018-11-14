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
from tensorflow.keras import utils, models, layers, Sequential, Model, callbacks

from src.constant import *


def gen_one():
    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    random_str = 'ajk8'
    X = generator.generate_image(random_str)
    X = np.expand_dims(X, 0)
    return X


def gen(batch_size=32):
    X = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, CHARACTERS_LENGTH), dtype=np.uint8) for i in range(CAPTCHA_LENGTH)]
    generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(CHARACTERS) for j in range(4)])
            # print(random_str)
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


# construct net
input_tensor = layers.Input((60, 160, 3))
x = input_tensor

for i in range(3):
    print(i, '---' * 30)
    x = layers.Convolution2D(32 * 2 ** i, (3, 3), activation='relu')(x)
    x = layers.Convolution2D(32 * 2 ** i, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dropout(0.25)(x)

x = [layers.Dense(CAPTCHA_LENGTH, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]

model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


class MyCallback(callbacks.Callback):

    def __init__(self):
        super().__init__()

    # def set_params(self, params):
    #     super().set_params(params)
    #
    # def set_model(self, model):
    #     super().set_model(model)
    #
    # def on_epoch_begin(self, epoch, logs=None):
    #     super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.model.save_weights('./weight/my_model', False)

    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)


# def evaluate(model, batch_num=10):
#     batch_acc = 0
#     generator = gen()
#     for i in range(batch_num):
#         [X_test, y_test, _, _], _ = next(generator)
#         y_pred = base_model.predict(X_test)
#         shape = y_pred[:, 2:, :].shape
#         ctc_decode = K.ctc_decode(y_pred[:, 2:, :],
#                                   input_length=np.ones(shape[0]) * shape[1])[0][0]
#         out = K.get_value(ctc_decode)[:, :4]
#         if out.shape[1] == 4:
#             batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
#     return batch_acc / batch_num
#
#
# class Evaluate(callbacks.Callback):
#
#     def __init__(self):
#         super().__init__()
#         self.accs = []
#
#     def on_epoch_end(self, epoch, logs=None):
#         acc = evaluate(base_model) * 100
#         self.accs.append(acc)
#         print('acc: %f%%' % acc)
#
#
# evaluator = Evaluate()

def train():
    my_callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        callbacks.TensorBoard(log_dir='./log'),
        # callbacks.ModelCheckpoint(),
        MyCallback(),
    ]

    model.fit_generator(generator=gen(), steps_per_epoch=100, epochs=5, callbacks=my_callbacks,
                        validation_data=gen(), validation_steps=20, workers=2)


def predict():
    model.load_weights('./weight/my_model')
    y = model.predict(gen_one())
    print(decode(y))


def show_network():
    utils.plot_model(model, to_file='model.png', show_shapes=True)
    pass


if __name__ == '__main__':
    # show_network()
    # X, y = next(gen(1))
    # plt.imshow(X[0])
    # plt.title(decode(y))
    # plt.show()
    # predict()
    pass
