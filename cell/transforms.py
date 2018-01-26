#-*- coding: utf8 -*-
import torch, numpy as np
import cv2, random, math
from PIL import Image
from torchvision.transforms import CenterCrop, Scale


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)

        return img1, img2


class NPToTensor(object):
    def __init__(self, norm=255):
        self.norm = norm

    def __call__(self, pic1, pic2):
        if not isinstance(pic1, np.ndarray):
            raise ValueError('Type must be np.ndarray')

        img1 = torch.from_numpy(pic1.transpose((2, 0, 1)).astype(np.float32)) # (B,G,R,NIR): [w,h,c] -> [c,w,h]
        img2 = torch.from_numpy(pic2.reshape(1, *pic2.shape).astype(np.float32))  # (B,G,R,NIR): [w,h,c] -> [c,w,h]

        if self.norm != 1:
            img1 = img1.div(self.norm) # -> [0,1]
            img2 = img2.div(self.norm)  # -> [0,1]

        return img1, img2


def resize(img1, img2, size):
    if isinstance(size, int):
        w, h, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img1, img2
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img1, (ow, oh)), cv2.resize(img2, (ow, oh))
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img1, (ow, oh)), cv2.resize(img2, (ow, oh))
    else:
        return cv2.resize(img1, size), cv2.resize(img2, size)


def random_shift_scale_rotate(img1, img2, p=0.75, shift_limit=16, scale_limit=0.1, rotate_limit=45):
    if random.random() < p:
        height,width,channel = img1.shape

        angle = random.uniform(-rotate_limit,rotate_limit)  # degree
        scale = random.uniform(1-scale_limit,1+scale_limit)

        dx = round(random.uniform(-shift_limit,shift_limit))  # pixel
        dy = round(random.uniform(-shift_limit,shift_limit))

        cc = math.cos(angle/180*math.pi) * scale
        ss = math.sin(angle/180*math.pi) * scale
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([[0,0], [width,0], [width,height], [0,height]])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)

        mat = cv2.getPerspectiveTransform(box0,box1)
        img1 = cv2.warpPerspective(img1, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        img2 = cv2.warpPerspective(img2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return img1, img2


def random_flip(img1, img2, p=0.5):
    if random.random() < p:
        x = random.randint(-1,1)
        img1 = cv2.flip(img1, x)
        img2 = cv2.flip(img2, x)
    return img1, img2


def random_transpose(img1, img2, p=0.5):
    if random.random() < p:
        img1 = cv2.transpose(img1)
        img2 = cv2.transpose(img2)
    return img1, img2