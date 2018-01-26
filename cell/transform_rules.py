#-*- coding: utf8 -*-
import sys
#import torchvision.transforms as transforms
from transforms import NPToTensor, resize, Compose, random_shift_scale_rotate, random_flip, random_transpose
from functools import partial


def np_nozoom_256():
    # TODO: add zoom
    # TODO: add mean, std

    train_transformations = Compose([
        #res
        partial(resize, size=(256, 256)),
        random_shift_scale_rotate,
        random_flip,
        random_transpose,
        NPToTensor(),
    ])

    val_transformations = Compose([
        NPToTensor(),
    ])

    test_transformation = Compose([
        #SpatialPick(),
        # TODO: average by augmentation
        NPToTensor(),
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def augmentation_factory(name):
    aug_func = globals().get(name, None)
    if aug_func is None:
        raise AttributeError("Model %s doesn't exist" % (name,))

    augm = aug_func()
    return augm


def main():
    pass


if __name__ == '__main__':
    sys.exit(main())