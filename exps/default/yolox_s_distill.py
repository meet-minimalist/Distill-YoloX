#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import ExpDistill as MyExpDistill


class Exp(MyExpDistill):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.temperature = 1.0
        self.kd_cls_weight = 0.5
        # self.kd_hint_weight = 0.5
        self.kd_hint_weight = 0.05
        self.pos_cls_weight = 1.0
        self.neg_cls_weight = 1.5
