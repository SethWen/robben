"""  
    author: Shawn
    time  : 11/8/18 6:00 PM
    desc  :
    update: Shawn 11/8/18 6:00 PM      
"""

import numpy as np
from src.constant import *


def char2pos(c):
    if c == '_':
        k = 62
        return k
    k = ord(c) - 48  # 0
    if k > 9:
        k = ord(c) - 55  # a
        if k > 35:
            k = ord(c) - 61
            if k > 61:
                raise ValueError('No Map')
    return k


def text2vec(text):
    text_len = len(text)
    if text_len > 4:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(CAPTCHA_LENGTH * CHARACTERS_LENGTH)

    for i, c in enumerate(text):
        idx = i * CHARACTERS_LENGTH + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHARACTERS_LENGTH
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


if __name__ == '__main__':
    a = text2vec('887o')
    print(vec2text(a))
