#-*- coding: utf8 -*-
import sys, os, glob, numpy as np
from datetime import datetime
import visdom, copy
from datetime import datetime
from collections import defaultdict
import cv2
from skimage.morphology import label
from skimage.filters import threshold_otsu

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomMonitor(object):
    def __init__(self, prefix=None, server='http://localhost', port=8097):
        self.__prefix = prefix or datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.__vis = visdom.Visdom(server=server, port=port)
        self.__metrics = defaultdict(lambda :defaultdict(list))
        self.__win_dict = {}
        self.__opts = self._init_opts()

    def _init_opts(self):
        opts = dict(legend=['Train', 'Validate'])
        return opts

    def __add(self, name, value, type):
        self.__metrics[type][name].append(value)

    def _add_val_performance(self, name, value):
        self.__add(name, value, type='val')

    def _add_train_performance(self, name, value):
        self.__add(name, value, type='train')

    def add_performance(self, metric_name, train_value, val_value):
        self._add_train_performance(metric_name, train_value )
        self._add_val_performance(metric_name, val_value)
        self.plot(metric_name)

    def plot(self, metric_name):
        current_win = self.__win_dict.get(metric_name, None)
        train_values = self.__metrics['train'][metric_name]
        val_values = self.__metrics['val'][metric_name]
        epochs = max(len(train_values), len(val_values))
        values_for_plot = np.column_stack((np.array(train_values), np.array(val_values)))
        opts = copy.deepcopy(self.__opts)
        opts.update(dict(title='%s\ntrain/val %s' % (self.__prefix, metric_name)))
        win = self.__vis.line(Y=values_for_plot, X=np.arange(epochs), opts=opts, win=current_win)

        if current_win is None:
            self.__win_dict[metric_name] = win


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, use_adaptive_threshold = False, cut_off = 0.5, connectivity = 2):
    if adaptive_threshold:
        cut_off = threshold_otsu(x)
    lab_img = label(x>cut_off, connectivity=connectivity)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)


def run_make_one_mask(data_dir):
    "Run on train dataset to make masks one"

    for example in os.listdir(data_dir):
        image_file = os.path.join(data_dir, example, 'images', example + '.png')
        print(example)
        mask_dir = os.path.join(data_dir, example, 'masks')
        target_dir = os.path.join(example, 'masks')
        mask_files = [os.path.join(data_dir, target_dir, file) for file in os.listdir(os.path.join(data_dir, target_dir))]
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        H,W,C = image.shape
        one_mask = np.zeros((H,W), dtype=bool)
        for mask_file in mask_files:
            mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
            one_mask = one_mask |(mask>128)

        one_mask = (one_mask*255).astype(np.uint8)
        out_dir = os.path.join(data_dir, example, 'mask')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = os.path.join(out_dir, 'one_mask.png')
        print (out_path)
        cv2.imwrite(out_path, one_mask)


def main():
    #from __init__ import TRAIN_FOLDER
    #run_make_one_mask(TRAIN_FOLDER)
    pass


if __name__ == '__main__':
    sys.exit(main())