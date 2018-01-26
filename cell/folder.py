#-*- coding: utf8 -*-
import os, sys
import cv2
import torch.utils.data as data
from torchvision.datasets.folder import default_loader


def jpg_cv2_loader(path):
    return cv2.imread(path,1)


def cv2_grayscale_loader(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


class ImageFolder(data.Dataset):
    def __init__(self, root, permitted_filenames=None, transform=None,
                 loader=jpg_cv2_loader):
        self.root = root
        self.imgs = self.make_dataset(permitted_filenames)
        self.transform = transform
        self.loader = loader

    def make_dataset(self, permitted_filenames):
        images = []
        for example in os.listdir(self.root):
            if permitted_filenames is None or example in permitted_filenames:
                image_path = os.path.join(example, 'images', example + '.png')
                target_path = os.path.join(example, 'mask', 'one_mask.png')
                images.append((image_path, target_path))

        return images

    def __getitem__(self, index):
        path, target_path = self.imgs[index]
        print (path)
        img = self.loader(os.path.join(self.root, path))
        target = cv2_grayscale_loader(os.path.join(self.root, target_path)) # TODO: replace
        #target = target.reshape(*target.shape, 1) # crutch

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageTestFolder(data.Dataset):
    def __init__(self, root, transform=None, loader=jpg_cv2_loader):
        imgs = [os.path.join(example,'images', example + '.png') for example in os.listdir(root)]
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img, _ = self.transform(img, img)

        return img, path

    def __len__(self):
        return len(self.imgs)


def main():
    import numpy as np
    from __init__ import TRAIN_FOLDER, TEST_FOLDER
    from transform_rules import augmentation_factory

    tr = augmentation_factory('np_nozoom_256')['train']

    #itf = ImageFolder(LABEL_FILE, TRAIN_FOLDER_TIF, loader=tif_loader)
    itf = ImageFolder(TRAIN_FOLDER, transform=tr)
    print(len(itf))
    for i, (img, target) in enumerate(itf):
        print (img.shape, target.shape)
        #cv2.imwrite('/home/tyantov/t1.png', img )
        #cv2.imwrite('/home/tyantov/t2.png', target)
        sys.exit()

    #print itf.classes
    #print itf.class_freq



if __name__ == '__main__':
    sys.exit(main())