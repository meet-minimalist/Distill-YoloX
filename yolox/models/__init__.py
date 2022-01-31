#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolox_loss import YoloXLoss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_head_vanilla import YOLOXHead as YOLOXHeadVanilla
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
from .yolox_wo_decoder import YOLOX as YOLOX_wo_Head
