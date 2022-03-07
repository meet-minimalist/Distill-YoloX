#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

# from .yolo_head import YOLOXHead
from .yolo_head_vanilla import YOLOXHead as YOLOXHeadVanilla
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, return_feats=False):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHeadVanilla(80)

        self.backbone = backbone
        self.head = head
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs, intermediate_featmap_outs = self.backbone(x)
        # fpn_outs[0] : [1, 128, 80, 80]
        # fpn_outs[1] : [1, 256, 40, 40]
        # fpn_outs[2] : [1, 512, 20, 20]

        # intermediate_featmap_outs will have features from different layers and with following keys
        # ["stem", "dark2", "dark3", "dark4", "dark5"]

        final_feature_maps = self.head(fpn_outs)
        if self.return_feats:
            return final_feature_maps, fpn_outs, intermediate_featmap_outs
        else:
            return final_feature_maps
        # return fpn_outs
        # if self.training:
        #     assert targets is not None
        #     loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
        #         fpn_outs, targets, x
        #     )
        #     outputs = {
        #         "total_loss": loss,
        #         "iou_loss": iou_loss,
        #         "l1_loss": l1_loss,
        #         "conf_loss": conf_loss,
        #         "cls_loss": cls_loss,
        #         "num_fg": num_fg,
        #     }
        # else:
        #     outputs = self.head(fpn_outs)

        # return outputs
