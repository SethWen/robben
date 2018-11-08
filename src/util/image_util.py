"""  
    author: Shawn
    time  : 11/8/18 6:11 PM
    desc  :
    update: Shawn 11/8/18 6:11 PM      
"""

import numpy as np
from PIL import Image


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def img2nparray(path):
    try:
        with Image.open(path) as image:
            return np.array(image)  # shape = (80, 170, 3) 3-RGB
    except Exception as e:
        print('img2nparray: error: ', e)
