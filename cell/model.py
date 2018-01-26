#-*- coding: utf8 -*-
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.densenet import densenet201
from torchvision.models.inception import inception_v3


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

def make_conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True),
    ]


class UNet(nn.Module):
    def __init__(self, pretrained, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        #256
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=2, padding=1 ),
        )
        #64
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32
        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=3, stride=1, padding=1 ),
        )
        #16
        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8
        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )
        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #16
        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32
        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64
        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        self.up0 = nn.Sequential(
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #256

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )

    def forward(self, x):
        down1 = self.down1(x) # 256
        out = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out = self.same(out)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #16
        out = torch.cat([down4, out],1)
        out = self.up4(out)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #32
        out = torch.cat([down3, out],1)
        out = self.up3(out)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #64
        out = torch.cat([down2, out],1)
        out = self.up2(out)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #128
        out = torch.cat([down1, out],1)
        out = self.up1(out)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #256
        out = self.up0(out)

        out = self.classify(out)
        out = F.sigmoid(out)

        return out


def unet(pretrained=False):
    model = UNet(pretrained=pretrained)
    return model


def factory(name='unet'):
    model_func = globals().get(name, None)
    if model_func is None:
        raise AttributeError("Model %s doesn't exist" % (name,))

    model = model_func() # pretrained=False
    return model


def main():
    model = factory('unet')
    from utils import dice_loss

    inputs = torch.randn(8, 3, 256, 256)
    labels = torch.LongTensor(8, 256, 256).random_(1).type(torch.FloatTensor)

    model = model.cuda().train()
    x = torch.autograd.Variable(inputs).cuda()
    y = torch.autograd.Variable(labels).cuda()
    logits = model.forward(x)

    loss = dice_loss(logits, y)
    loss.backward()

    print(type(model))
    print(model)

    print('logits')
    print(logits)

if __name__ == '__main__':
    sys.exit(main())