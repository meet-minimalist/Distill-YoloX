# Created Date: Sunday, January 30th 2022, 9:33:15 am
# Author: meet_minimalist
# Copyright (c) 2022 

from lib2to3.pytree import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class KDLoss_SA(nn.Module):
    def __init__(self, student_device='cuda:0', teacher_device='cuda:0'):
        super(KDLoss_SA, self).__init__()

        self.l1_loss = nn.L1Loss()
        self.student_device = student_device
        self.teacher_device = teacher_device


    def __feat_transform(self, feat):
        # Ref. : https://github.com/Vincent-Hoo/Knowledge-Distillation-for-Super-resolution/blob/ff673aee08eca674925527e7c90b13b389eee9a0/code/feature_transformation.py#L4

        assert len(feat.shape) == 4
        # feat : [N, C, H, W]

        # Computing matmuls on larger dimensions of H and W consumes lots of GPU resources. 
        # We can apply average pool over the feat map to make all the feature map 10x10 from 160x160, 80x80, 40x40 and 20x20.
        # Then we apply our logic of normalization on top of that.

        assert feat.shape[2] == feat.shape[3]

        k_size = feat.shape[2] // 10
        stride = k_size
        feat = nn.AvgPool2d(k_size, stride)(feat)
        
        feat = feat.view(feat.size(0), feat.size(1), -1)
        # feat : [N, C, H*W]
        norm_feat = feat / (torch.sqrt(torch.sum(torch.pow(feat,2), 1)).unsqueeze(1).expand(feat.shape) + 0.0000001)
        s = norm_feat.transpose(1,2).bmm(norm_feat)
        # s : [N, H*W, H*W]
        s = s.unsqueeze(1)
        # s : [N, 1, H*W, H*W]
        return s


    def forward(self, student_backbone_feat, teacher_backbone_feat, \
                    student_fpn_feat, teacher_fpn_feat):
        # student_backbone_feat : it will have features from different layers and with following keys ["stem", "dark2", "dark3", "dark4", "dark5"]
        # teacher_backbone_feat : same as above
        # dark2 : [1, 64, 160, 160] or [1, 128, 160, 160]
        # dark2 : [1, 128, 80, 80] or [1, 256, 80, 80]
        # dark2 : [1, 256, 40, 40] or [1, 512, 40, 40]
        # dark2 : [1, 512, 20, 20] or [1, 1024, 20, 20]

        # student_fpn_feat : a list of 3 feature maps having shapes [1, 128, 80, 80], [1, 256, 40, 40], [1, 512, 20, 20]
        # teacher_fpn_feat : same as above but with [1, 256, 80, 80], [1, 512, 40, 40], [1, 1024, 20, 20]
        

        loss_sa = 0
        for (s_k, s_v), (t_k, t_v) in zip(student_backbone_feat.items(), teacher_backbone_feat.items()):
            if s_k not in ['stem'] and t_k not in ['stem']:
                s_v = self.__feat_transform(s_v)
                t_v = self.__feat_transform(t_v)
                loss_sa += self.l1_loss(s_v, t_v)
        
        for (s_v, t_v) in zip(student_fpn_feat, teacher_fpn_feat):
            s_v = self.__feat_transform(s_v)
            t_v = self.__feat_transform(t_v)
            loss_sa += self.l1_loss(s_v, t_v)

        return loss_sa

