#-*- coding: utf8 -*-
import os
import logging

__all__ = ['PACKAGE_DIR', 'RESULT_DIR', 'DATA_FOLDER', 'TRAIN_FOLDER', 'TEST_FOLDER']


PACKAGE_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')
RESULT_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, '../', 'results'))
DATA_FOLDER = os.path.realpath(os.path.join(PACKAGE_DIR, '../data'))
TRAIN_FOLDER = os.path.join(DATA_FOLDER, 'stage1_train')
TEST_FOLDER = os.path.join(DATA_FOLDER, 'stage1_test')

# SPLIT_FILE = os.path.join(data_folder, 'split.txt')
# HOLDOUT_FILE = os.path.join(data_folder, 'holdout.txt')

assert os.path.exists(DATA_FOLDER)
assert os.path.exists(TRAIN_FOLDER)
assert os.path.exists(TEST_FOLDER)

if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

# logging
logger = logging.getLogger('app')
fh = logging.FileHandler(os.path.join(RESULT_DIR, 'application.log'))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

#Test
#import torch
#print torch.has_cudnn
