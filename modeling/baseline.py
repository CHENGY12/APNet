
import torch
from torch import nn
import torch.nn.functional as F
import sys
import pdb

from .backbones.se_module import SELayer
from .backbones.inception import BasicConv2d
from .backbones.resnet import ResNet
from .backbones.resnest import resnest50
sys.path.append('.')


EPSILON = 1e-12


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class SAMS(nn.Module):
    """
    Split-Attend-Merge-Stack agent
    Input an feature map with shape H*W*C, we first split the feature maps into
    multiple parts, obtain the attention map of each part, and the attention map
    for the current pyramid level is constructed by mergiing each attention map.
    """
    def __init__(self, in_channels, channels,
                 radix=4, reduction_factor=4,
                norm_layer=nn.BatchNorm2d):
        super(SAMS, self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=1)


    def forward(self, x):

        batch, channel = x.shape[:2]
        splited = torch.split(x, channel//self.radix, dim=1)

        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, channel//self.radix, dim=1)

        out= torch.cat([att*split for (att, split) in zip(atten, splited)],1)
        return out.contiguous()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return  y

class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path,level):
        super(Baseline, self).__init__()
        print(f"Training with pyramid level {level}")
        self.level = level
        self.base = ResNet(last_stride= last_stride)

        self.base.load_param(model_path)
        self.base_1 = nn.Sequential(*list(self.base.children())[0:3])
        self.base_2 = nn.Sequential(*list(self.base.children())[3:4])
        self.base_3 = nn.Sequential(*list(self.base.children())[4:5])
        self.base_4 = nn.Sequential(*list(self.base.children())[5:6])
        self.base_5 = nn.Sequential(*list(self.base.children())[6:])


        if self.level > 0:
            self.att1 = SELayer(64,8)
            self.att2 = SELayer(256,32)
            self.att3 = SELayer(512,64)
            self.att4 = SELayer(1024,128)
            self.att5 = SELayer(2048,256)
            if self.level > 1: # second pyramid level
                self.att_s1=SAMS(64,int(64/self.level),radix=self.level)
                self.att_s2=SAMS(256,int(256/self.level),radix=self.level)
                self.att_s3=SAMS(512,int(512/self.level),radix=self.level)
                self.att_s4=SAMS(1024,int(1024/self.level),radix=self.level)
                self.att_s5=SAMS(2048,int(2048/self.level),radix=self.level)
                self.BN1 = BN2d(64)
                self.BN2 = BN2d(256)
                self.BN3 = BN2d(512)
                self.BN4 = BN2d(1024)
                self.BN5 = BN2d(2048)

                if self.level > 2:
                    self.att_ss1=SAMS(64,int(64/self.level),radix=self.level)
                    self.att_ss2=SAMS(256,int(256/self.level),radix=self.level)
                    self.att_ss3=SAMS(512,int(512/self.level),radix=self.level)
                    self.att_ss4=SAMS(1024,int(1024/self.level),radix=self.level)
                    self.att_ss5=SAMS(2048,int(2048/self.level),radix=self.level)
                    self.BN_1 = BN2d(64)
                    self.BN_2 = BN2d(256)
                    self.BN_3 = BN2d(512)
                    self.BN_4 = BN2d(1024)
                    self.BN_5 = BN2d(2048)
                    if self.level > 3:
                        raise RuntimeError("We do not support pyramid level greater than three.")

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):


        #pdb.set_trace()
        x = self.base_1(x)
        if self.level > 2:
            x = self.att_ss1(x)
            x = self.BN_1(x)
        if self.level > 1:
            x = self.att_s1(x)
            x = self.BN1(x)
        if self.level > 0:
            y = self.att1(x)
            x=x*y.expand_as(x)


        x = self.base_2(x)
        if self.level > 2:
            x = self.att_ss2(x)
            x = self.BN_2(x)
        if self.level > 1:
            x = self.att_s2(x)
            x = self.BN2(x)
        if self.level > 0:
            y = self.att2(x)
            x=x*y.expand_as(x)


        x = self.base_3(x)
        if self.level > 2:
            x = self.att_ss3(x)
            x = self.BN_3(x)
        if self.level > 1:
            x = self.att_s3(x)
            x = self.BN3(x)
        if self.level > 0:
            y = self.att3(x)
            x=x*y.expand_as(x)

        x = self.base_4(x)
        if self.level > 2:
            x = self.att_ss4(x)
            x = self.BN_4(x)
        if self.level > 1:
            x = self.att_s4(x)
            x = self.BN4(x)
        if self.level > 0:
            y = self.att4(x)
            x=x*y.expand_as(x)


        x = self.base_5(x)
        if self.level > 2:
            x = self.att_ss5(x)
            x = self.BN_5(x)
        if self.level > 1:
            x = self.att_s5(x)
            x = self.BN5(x)
        if self.level > 0:
            y = self.att5(x)
            x=x*y.expand_as(x)


        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            return feat

            # return self.classifier(feat)
