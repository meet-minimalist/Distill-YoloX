#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .base_exp import BaseExp
from .build import get_exp
from .yolox_base import Exp
from .yolox_base_voc import ExpVOC
from .yolox_base_distill import Exp as ExpDistill
from .yolox_base_distill_voc import ExpVOC as ExpVOCDistill
