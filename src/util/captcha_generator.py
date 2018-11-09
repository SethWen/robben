"""  
    author: Shawn
    time  : 11/7/18 4:18 PM
    desc  :
    update: Shawn 11/7/18 4:18 PM      
"""

from src import constant
from captcha.image import ImageCaptcha
import random
import time


def gen_one():
    """
    generate captcha and save it to current path
    :return:
    """
    generator = ImageCaptcha(width=constant.IMAGE_WIDTH, height=constant.IMAGE_HEIGHT)
    random_str = ''.join([random.choice(constant.CHARACTERS) for j in range(4)])
    img = generator.generate_image(random_str)
    # img.save('/data/gen/{}.png'.format(random_str))
    stamp = '{:.0f}'.format(time.time() * 1000)
    img.save('./trainset/{}_{}.png'.format(stamp, random_str))


if __name__ == '__main__':
    # gen_one()
    i = 0
    while i < 3000:
        gen_one()
        i = i + 1

    pass
