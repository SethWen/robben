"""
    author: Shawn
    time  : 11/8/18 7:11 PM
    desc  :
    update: Shawn 11/8/18 7:11 PM      
"""

import sys
import os
from src.core import trainer

# add cannes path to python path
sys.path.append(os.getcwd())

if __name__ == '__main__':
    trainer.train_through_cnn()
