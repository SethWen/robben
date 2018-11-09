"""
    author: Shawn
    time  : 11/8/18 7:11 PM
    desc  :
    update: Shawn 11/8/18 7:11 PM      
"""

import sys
import os
from src.core import trainer1
from src.util import text_util, image_util, data_generator

# add cannes path to python path
# `export PYTHONPATH=$PATHONPATH:~/WorkSpace/shawn/robben`
sys.path.append(os.getcwd())

if __name__ == '__main__':
    # train
    # trainer1.train_through_cnn()

    # test
    # text, image = data_generator.gen_captcha_text_and_image()
    # image = image_util.convert2gray(image)
    # image = image.flatten() / 255
    # # print(image.shape)
    # predict_text = trainer1.predict(image)
    # print("正确: {}  预测: {}".format(text, predict_text))
    pass
