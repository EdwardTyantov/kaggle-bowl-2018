#-*- coding: utf8 -*-
import sys, cv2
from torchvision.transforms import ToTensor, Compose as SingleCompose
from transforms import NPToTensor, resize, Compose, random_shift_scale_rotate, random_flip, random_transpose, \
    resize_single, SpatialPick, zoom, channel_shift, double_channel_shift
from functools import partial


def np_nozoom_256():
    # TODO: add mean, std

    train_transformations = Compose([
        #res
        partial(resize, size=(192, 192)), # TODO: not to resize
        random_shift_scale_rotate,
        random_flip,
        random_transpose,
        #partial(double_channel_shift, intensity=10),
        NPToTensor(),
    ])

    val_transformations = Compose([
        partial(resize, size=(192, 192)),
        NPToTensor(),
    ])

    test_transformation = SingleCompose([     # solo transform
        partial(resize_single, size=(192, 192)),
        SpatialPick((
            {'do': lambda x: x, 'undo': lambda x: x},
            {'do': partial(cv2.flip, flipCode=-1), 'undo': partial(cv2.flip, flipCode=-1)},
            {'do': partial(cv2.flip, flipCode=0), 'undo': partial(cv2.flip, flipCode=0)},
            {'do': partial(cv2.flip, flipCode=1), 'undo': partial(cv2.flip, flipCode=1)},
            # {'do': partial(zoom, scale=1.05), 'undo': partial(zoom, scale=1/1.05)},
            {'do': partial(zoom, scale=0.95), 'undo': partial(zoom, scale=1/0.95)},
            #{'do': partial(channel_shift, intensity=10), 'undo': lambda x: x},
             )),
        ToTensor(),
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