# Created Date: Sunday, January 30th 2022, 9:33:15 am
# Author: meet_minimalist
# Copyright (c) 2022 

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class KDLoss(nn.Module):
    def __init__(self, temperature=1.0, kd_cls_weight=0.5, kd_hint_weight=0.5, \
                    pos_cls_weight=1.0, neg_cls_weight=1.5, student_device='cuda:0', \
                    teacher_device='cuda:0'):
        super(KDLoss, self).__init__()

        self.mse = nn.MSELoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)
        self.temperature = temperature
        self.student_device = student_device
        self.teacher_device=  teacher_device

        self.kd_cls_weight = kd_cls_weight
        self.kd_hint_weight = kd_hint_weight
        self.pos_w = pos_cls_weight
        self.neg_w = neg_cls_weight


    def forward(self, student_feat_map, teacher_feat_map):
        # student_feat_map : a dict with key 0, 1, 2 and respective feat maps
        # teacher_feat_map : a dict with key 0, 1, 2 and respective feat maps

        # 0 : [B x 4 x 80 x 80], [B x 1 x 80 x 80], [B x 80 x 80 x 80]
        # 1 : [B x 4 x 40 x 40], [B x 1 x 40 x 40], [B x 80 x 40 x 40]
        # 2 : [B x 4 x 20 x 20], [B x 1 x 20 x 20], [B x 80 x 20 x 20]


        loss_kd_hint = 0
        loss_kd_softmax_temp = 0
        for (student_fmap_op, teacher_fmap_op) in zip(student_feat_map.values(), teacher_feat_map.values()):
            student_reg_output, student_obj_output, student_cls_output = student_fmap_op
            teacher_reg_output, teacher_obj_output, teacher_cls_output = teacher_fmap_op
            
            # reg, obj, cls : [B x 4 x 20 x 20], [B x 1 x 20 x 20], [B x 80 x 20 x 20]
            if self.teacher_device != self.student_device:
                teacher_reg_output = teacher_reg_output.to(self.student_device)
                teacher_obj_output = teacher_obj_output.to(self.student_device)
                teacher_cls_output = teacher_cls_output.to(self.student_device)

            # TODO : Normalize each mse loss by feature-map height width then 
            #      : take the sum across all the scales.
            feat_h, feat_w = student_reg_output.shape[2:4]

            normalized_hint_loss = torch.sum(self.mse(student_reg_output, teacher_reg_output) / (feat_h * feat_w), dim=[1, 2, 3]) + \
                                   torch.sum(self.mse(student_obj_output, teacher_obj_output) / (feat_h * feat_w), dim=[1, 2, 3]) + \
                                   torch.sum(self.mse(student_cls_output, teacher_cls_output) / (feat_h * feat_w), dim=[1, 2, 3])

            loss_kd_hint += torch.mean(normalized_hint_loss)

            conf = self.softmax(student_cls_output/self.temperature)
            conf_k = self.softmax(teacher_cls_output/self.temperature)
            loss_kd_softmax_temp += self.weighted_kl_div(conf, conf_k)

        loss_kd_hint = loss_kd_hint * self.kd_hint_weight
        return loss_kd_softmax_temp, loss_kd_hint


    def weighted_kl_div(self, ps, qt):
        # ps, qt shape : [B, C, H, W]
        eps = 1e-10
        ps = ps + eps
        qt = qt + eps
        log_p = qt * torch.log(ps)
        log_p[:, 0] *= self.neg_w
        log_p[:, 1:] *= self.pos_w

        # return -torch.sum(log_p)
        return torch.mean(-torch.sum(log_p, dim=1))

