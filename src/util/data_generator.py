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
    # print(files)  # 当前路径下所有非目录子文件
    return root, files


def get_batch_image(batch=64):
    """
    generate data from images
    :param batch:
    :return:
    """
    root, files = list_files(constant.IMAGE_PATH)
    batch_x = np.zeros([batch, constant.IMAGE_WIDTH * constant.IMAGE_HEIGHT])
    batch_y = np.zeros([batch, constant.CAPTCHA_LENGTH * constant.CHARACTERS_LENGTH])

    for f in files:
        for i in range(batch):
            image = image_util.img2nparray(os.path.join(root, f))
            image = image_util.convert2gray(image)
            label = f.replace('./', '').replace('.png', '')
            batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
            batch_y[i, :] = text_util.text2vec(label)
        yield batch_x, batch_y


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
    return captcha_text, captcha_image


if __name__ == '__main__':
    # j = 0
    # for x, y in get_batch_image(2):
    #     print('-------------------------' * 5)
    #     print('x.shape = ', x.shape)
    #     print('y.shape = ', y.shape)
    #     j += 1
    #     if j == 10:
    #         break

    # print(random_captcha_text())
    # print(gen_captcha_text_and_image()[1].shape)
    pass
