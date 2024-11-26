from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn

from model.posenet import posenet
from model.segnet import segnet


class Res_catconv(nn.Module):
    def __init__(self, segnet, posenet):
        super(Res_catconv, self).__init__()
        self.segnet = segnet
        self.posenet = posenet
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, data):
        img = data['img']
        partialheat = self.posenet(img)
        pre_mask = self.segnet(img, partialheat)

        input_heat = data['gt_heat'][-1]

        return dict(mask=pre_mask[-1], heatmap=input_heat, premask=pre_mask, preheat=partialheat)

# create network
def poseseg(generator=None):
    pose = posenet()
    seg = segnet()
    UV_net = Res_catconv(seg, pose)
    return UV_net
