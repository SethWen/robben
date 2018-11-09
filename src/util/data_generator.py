"""  
    author: Shawn
    time  : 11/7/18 4:58 PM
    desc  :
    update: Shawn 11/7/18 4:58 PM      
"""

import os
import random
import numpy as np
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image
from src.util import text_util, image_util
from src import constant


def list_files(file_dir):
    if not os.path.isabs(file_dir):
        raise Exception('must be absolute path')

    root, dirs, files = next(os.walk(file_dir))
    # print(root)  # 当前目录路径
    # print(dirs)  # 当前路径下所有子目录
    # print(len(files))  # 当前路径下所有非目录子文件
    for f in files:
        image = image_util.img2nparray(os.path.join(root, f))
        image = image_util.convert2gray(image)
        image = image.flatten() / 255  # (image.flatten()-128)/128  mean为0

        start = f.find('_') + 1
        end = f.rfind('.')
        label = f[start:end]
        yield image, label


def get_batch_image(img_gen, batch=64):
    """
    generate data from images
    :param img_gen:
    :param batch:
    :return:
    """
    batch_x = np.zeros([batch, constant.IMAGE_WIDTH * constant.IMAGE_HEIGHT])
    batch_y = np.zeros([batch, constant.CAPTCHA_LENGTH * constant.CHARACTERS_LENGTH])

    for i in range(batch):
        image, label = next(img_gen)
        batch_x[i, :] = image
        batch_y[i, :] = text_util.text2vec(label)

    return batch_x, batch_y


def random_captcha_text(char_set=constant.CHARACTERS, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    """
    generate random captcha
    :return: (captcha text, np.array of the captcha)
    """
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    print(captcha_text)
    return captcha_text, captcha_image


def test():
    epoch = 0
    step = 0
    gen = list_files(constant.IMAGE_PATH)
    while epoch < 5:  # epoch
        try:
            data = get_batch_image(gen, 128)
        except StopIteration as e:
            epoch += 1
            gen = list_files(constant.IMAGE_PATH)
            data = get_batch_image(gen, 128)
            pass

        print(data[0].shape, data[1].shape)  # (128, 9600) (128, 16)
        step += 1


if __name__ == '__main__':
    test()
    pass
