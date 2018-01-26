#-*- coding: utf8 -*-
import sys
#import torchvision.transforms as transforms
from transforms import NPToTensor, resize, Compose, random_shift_scale_rotate, random_flip, random_transpose
from functools import partial


def np_nozoom_256():
    # TODO: add zoom
    # TODO: add mean, std
    # TODO: add random_channel_shift

    train_transformations = Compose([
        #res
        partial(resize, size=(256, 256)), # TODO: not to resize
        random_shift_scale_rotate,
        random_flip,
        random_transpose,
        NPToTensor(),
    ])

    val_transformations = Compose([
        partial(resize, size=(256, 256)),
        NPToTensor(),
    ])

    test_transformation = Compose([     # solo transform
        partial(resize, size=(256, 256)),
        #SpatialPick(),
        # TODO: average by augmentation
        NPToTensor(),
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}


def augmentation_factory(name):
    aug_func = globals().get(name, None)
    if aug_func is None:
        raise AttributeError("Augmentation %s doesn't exist" % (name,))

    augm = aug_func()
    return augm


def main():
    pass


if __name__ == '__main__':
    sys.exit(main())