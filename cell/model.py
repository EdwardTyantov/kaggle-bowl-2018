#-*- coding: utf8 -*-
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
#from torchvision.models.densenet import densenet201
from torchvision.models.inception import inception_v3, BasicConv2d


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


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, bn=False, activation=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.__bn = bn
        self.__act = activation

    def forward(self, x):
        x = self.conv(x)
        if self.__bn:
            x = self.bn(x)
        if self.__act:
            x = F.elu(x, inplace=True)
        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, depth):
        super(InceptionBlock, self).__init__()
        # TODO add elu
        assert depth % 16 == 0

        self.c1_1 = BasicConv2d(in_channels, depth // 4, bn=False, kernel_size=1, stride=1, padding=0)

        self.c2_1 = BasicConv2d(in_channels, depth // 8 * 3, bn=False, activation=True, kernel_size=1, stride=1, padding=0)
        self.c2_2 = BasicConv2d(depth // 8 * 3, depth // 2, bn=True, activation=True, kernel_size=(1,3), stride=1, padding=(0,1))
        self.c2_3 = BasicConv2d(depth // 2, depth // 2, bn=False, activation=False, kernel_size=(3,1), stride=1, padding=(1,0))

        self.c3_1 = BasicConv2d(in_channels, depth // 16, bn=False, activation=True, kernel_size=1, stride=1, padding=0)
        self.c3_2 = BasicConv2d(depth // 16, depth // 8, bn=True, activation=True, kernel_size=(1,5), stride=1, padding=(0,2))
        self.c3_3 = BasicConv2d(depth // 8, depth // 8, bn=False, activation=False, kernel_size=(5,1), stride=1, padding=(2,0))

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.c4_2 = BasicConv2d(in_channels, depth // 8, bn=False, activation=False, kernel_size=1, stride=1, padding=0)

        self.final_bn = nn.BatchNorm2d(depth, eps=0.001)

    def forward(self, x):
        c1_1 = self.c1_1(x)

        c2_1 = self.c2_1(x)
        c2_2 = self.c2_2(c2_1)
        c2_3 = self.c2_3(c2_2)

        c3_1 = self.c3_1(x)
        c3_2 = self.c3_2(c3_1)
        c3_3 = self.c3_3(c3_2)

        p4_1 = self.p4_1(x)
        c4_2 = self.c4_2(p4_1)

        link = torch.cat([c1_1, c2_3, c3_3, c4_2], 1)
        out = self.final_bn(link)
        out = F.elu(out, inplace=True)

        return out


class RBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=0.1):
        super(RBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channel, out_channel, bn=True, activation=True, kernel_size=1, stride=1, padding=0)
        self.__scale = scale

    def forward(self, x):
        residual = x * self.__scale
        out = self.conv1(x)

        out = out + residual
        out = F.elu(out, inplace=True)

        return out


class MyNet(nn.Module):
    def __init__(self, pretrained=False, in_channels=3, num_classes=1):
        super(MyNet, self).__init__()
        #256
        self.down1 = InceptionBlock(3, 32)
        self.pool1 = BasicConv2d(32, 32, bn=True, activation=True, kernel_size=3, stride=2, padding=1)
        self.drop1 = nn.Dropout2d(p=0.5, inplace=False)

        self.down2 = InceptionBlock(32, 64)
        self.pool2 = BasicConv2d(64, 64, bn=True, activation=True, kernel_size=3, stride=2, padding=1)
        self.drop2 = nn.Dropout2d(p=0.5, inplace=False)

        self.down3 = InceptionBlock(64, 128)
        self.pool3 = BasicConv2d(128, 128, bn=True, activation=True, kernel_size=3, stride=2, padding=1)
        self.drop3 = nn.Dropout2d(p=0.5, inplace=False)

        self.down4 = InceptionBlock(128, 256)
        self.pool4 = BasicConv2d(256, 256, bn=True, activation=True, kernel_size=3, stride=2, padding=1)
        self.drop4 = nn.Dropout2d(p=0.5, inplace=False)

        self.down5 = InceptionBlock(256, 512)
        self.drop5 = nn.Dropout2d(p=0.5, inplace=False)

        # TODO: insert here second branch

        self.after_conv4 = RBlock(256, 256)
        self.conv6 = InceptionBlock(768, 256)
        self.drop6 = nn.Dropout2d(p=0.5, inplace=False)

        self.after_conv3 = RBlock(128, 128)
        self.conv7 = InceptionBlock(384, 128)
        self.drop7 = nn.Dropout2d(p=0.5, inplace=False)

        self.after_conv2 = RBlock(64, 64)
        self.conv8 = InceptionBlock(192, 64)
        self.drop8 = nn.Dropout2d(p=0.5, inplace=False)

        self.after_conv1 = RBlock(32, 32)
        self.conv9 = InceptionBlock(96, 32)
        self.drop9 = nn.Dropout2d(p=0.5, inplace=False)

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )

    def forward(self, x):
        down1 = self.down1(x) # 256
        #out = self.drop1(self.pool1(down1))
        out = self.pool1(down1)

        down2 = self.down2(out)
        #out = self.drop2(self.pool2(down2))
        out = self.pool2(down2)

        down3 = self.down3(out)
        #out = self.drop3(self.pool3(down3))
        out = self.pool3(down3)

        down4 = self.down4(out)
        #out = self.drop4(self.pool4(down4))
        out = self.pool4(down4)

        down5 = self.down5(out)
        #out = self.drop5(down5)
        out = down5

        # here second branch

        after_conv4 = self.after_conv4(down4)

        up6 = F.upsample(out, scale_factor=2, mode='bilinear')
        up6 = torch.cat([up6, after_conv4], 1)
        #out = self.drop6(self.conv6(up6))
        out = self.conv6(up6)

        after_conv3 = self.after_conv3(down3)
        up7 = F.upsample(out, scale_factor=2, mode='bilinear')
        up7 = torch.cat([up7, after_conv3], 1)
        #out = self.drop7(self.conv7(up7))
        out = self.conv7(up7)

        after_conv2 = self.after_conv2(down2)
        up8 = F.upsample(out, scale_factor=2, mode='bilinear')
        up8 = torch.cat([up8, after_conv2], 1)
        #out = self.drop8(self.conv8(up8))
        out = self.conv8(up8)

        after_conv1 = self.after_conv1(down1)
        up9 = F.upsample(out, scale_factor=2, mode='bilinear')
        up9 = torch.cat([up9, after_conv1], 1)
        #out = self.drop8(self.conv9(up9))
        out = self.conv9(up9)

        out = self.classify(out)
        out = F.sigmoid(out)

        return out


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
        # out = F.dropout(out, training=self.training)

        down2 = self.down2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2) #32
        # out = F.dropout(out, training=self.training)

        down3 = self.down3(out)
        out = F.max_pool2d(down3, kernel_size=2, stride=2) #16
        # out = F.dropout(out, training=self.training)

        down4 = self.down4(out)
        out = F.max_pool2d(down4, kernel_size=2, stride=2) # 8
        # out = F.dropout(out, training=self.training)

        out = self.same(out)
        # out = F.dropout(out, training=self.training)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #16
        out = torch.cat([down4, out],1)
        out = self.up4(out)
        # out = F.dropout(out, training=self.training)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #32
        out = torch.cat([down3, out],1)
        out = self.up3(out)
        # out = F.dropout(out, training=self.training)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #64
        out = torch.cat([down2, out],1)
        out = self.up2(out)
        # out = F.dropout(out, training=self.training)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #128
        out = torch.cat([down1, out],1)
        out = self.up1(out)
        # out = F.dropout(out, training=self.training)

        out = F.upsample(out, scale_factor=2, mode='bilinear') #256
        out = self.up0(out)
        # out = F.dropout(out, training=self.training)

        out = self.classify(out)
        out = F.sigmoid(out)

        return out


def unet(pretrained=False):
    model = UNet(pretrained=pretrained)
    return model

def mynet(pretrained=False):
    model = MyNet(pretrained=pretrained)
    return model

def factory(name='unet'):
    model_func = globals().get(name, None)
    if model_func is None:
        raise AttributeError("Model %s doesn't exist" % (name,))

    model = model_func() # pretrained=False
    return model


def main():
    model = factory('mynet')
    from utils import dice_loss

    inputs = torch.randn(8, 3, 256, 256)
    labels = torch.LongTensor(8, 256, 256).random_(1).type(torch.FloatTensor)

    model = model.train()
    print (model)
    # model = InceptionA(3, 64)
    # print (model)
    # sys.exit()

    x = torch.autograd.Variable(inputs)
    y = torch.autograd.Variable(labels)
    logits = model.forward(x)

    loss = dice_loss(logits, y)
    loss.backward()

    print(type(model))
    print(model)

    print('logits')
    print(logits)

if __name__ == '__main__':
    sys.exit(main())