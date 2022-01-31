# Created Date: Sunday, January 30th 2022, 9:33:15 am
# Author: meet_minimalist
# Copyright (c) 2022 

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from yolox.utils import bboxes_iou

from .losses import IOUloss


class YoloXLoss(nn.Module):
    def __init__(self, kd_loss=True, temperature=1.0, kd_cls_weight=0.5, kd_hint_weight=0.5, strides=[8, 16, 32]):
        super(YoloXLoss, self).__init__()
        self.kd_loss = kd_loss
        
        self.use_l1 = False
        # Note during training this will be set as True externally from trainer_distill.py file.

        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.mse = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)
        self.temperature = temperature
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)


        self.kd_cls_weight = kd_cls_weight
        self.kd_hint_weight = kd_hint_weight

    def forward(self, feat_maps, labels):
        if self.kd_loss:
            student_feat_map, teacher_feat_map = feat_maps
            # student_feat_map : a dict with key 0, 1, 2 and respective feat maps
            # teacher_feat_map : a dict with key 0, 1, 2 and respective feat maps

            # 0 : [B x 4 x 80 x 80], [B x 1 x 80 x 80], [B x 80 x 80 x 80]
            # 1 : [B x 4 x 40 x 40], [B x 1 x 40 x 40], [B x 80 x 40 x 40]
            # 2 : [B x 4 x 20 x 20], [B x 1 x 20 x 20], [B x 80 x 20 x 20]


            loss_iou, loss_obj, loss_cls, loss_l1, num_fg = self.yolo_loss(student_feat_map, labels)
        
            loss_kd_hint = 0
            loss_kd_softmax_temp = 0
            for (student_fmap_op, teacher_fmap_op) in zip(student_feat_map.values(), teacher_feat_map.values()):
                student_reg_output, student_obj_output, student_cls_output = student_fmap_op
                teacher_reg_output, teacher_obj_output, teacher_cls_output = teacher_fmap_op

                loss_kd_hint += self.mse(student_reg_output, teacher_reg_output) + \
                                self.mse(student_obj_output, teacher_obj_output) + \
                                self.mse(student_cls_output, teacher_cls_output)

                conf = self.softmax(student_cls_output/self.temperature)
                conf_k = self.softmax(teacher_cls_output/self.temperature)
                loss_kd_softmax_temp += self.weighted_kl_div(conf, conf_k)
                
            loss_cls_kd = self.kd_cls_weight * loss_kd_softmax_temp
            loss_cls = (1 - self.kd_cls_weight) * loss_cls
            loss_cls_total = loss_cls + loss_cls_kd

            reg_weight = 5.0
            loss_iou = reg_weight * loss_iou

            loss_kd_hint = loss_kd_hint * self.kd_hint_weight

            loss_total = loss_iou + loss_obj + loss_cls_total + loss_l1 + loss_kd_hint


        else:
            model_feat_map = feat_maps[0]
            # model_feat_map : a dict with key 0, 1, 2 and respective feat maps

            loss_iou, loss_obj, loss_cls, loss_l1, num_fg = self.yolo_loss(model_feat_map, labels)

            reg_weight = 5.0
            loss_iou = reg_weight * loss_iou
            loss_total = loss_iou + loss_obj + loss_cls + loss_l1
            loss_cls_kd = 0.0
            loss_kd_hint = 0.0

        return loss_total, loss_iou, loss_obj, loss_cls, loss_cls_kd, loss_l1, loss_kd_hint, num_fg
            

    def yolo_loss(self, model_feat_map, labels):
        # model_feat_map : a dict with key 0, 1, 2 and respective feat maps
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, (reg_output, obj_output, cls_output) in model_feat_map.items():    
            stride_this_level = self.strides[k]
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(
                output, k, stride_this_level, reg_output[0].type()
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type_as(reg_output[0])
            )
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, self.n_anchors, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())

            outputs.append(output)


        loss_iou, loss_obj, loss_cls, loss_l1, num_fg = self.get_losses(
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            torch.cat(outputs, 1),
            origin_preds,
            dtype=reg_output[0].dtype,
        )

        return loss_iou, loss_obj, loss_cls, loss_l1, num_fg

    def weighted_kl_div(self, ps, qt):
        eps = 1e-10
        ps = ps + eps
        qt = qt + eps
        log_p = qt * torch.log(ps)
        log_p[:, 0] *= self.neg_w
        log_p[:, 1:] *= self.pos_w
        return -torch.sum(log_p)


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid
    
    
    def get_losses(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        # reg_weight = 5.0
        # loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            # loss,
            loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )


    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
