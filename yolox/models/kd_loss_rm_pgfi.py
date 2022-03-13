# Created Date: Sunday, January 30th 2022, 9:33:15 am
# Author: meet_minimalist
# Copyright (c) 2022 

from lib2to3.pytree import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class KDLoss_RM_PGFI(nn.Module):
    def __init__(self, student_channels, teacher_channels, \
                    student_device='cuda:0', teacher_device='cuda:0', \
                    in_channels=[256, 512, 1024], num_classes=80, strides=[8, 16, 32]):
        super(KDLoss_RM_PGFI, self).__init__()

        self.mse = nn.MSELoss()
        self.mse_no_red = nn.MSELoss(reduction='none')
        self.softmax_d1 = nn.Softmax(dim=1)
        self.softmax_d0 = nn.Softmax(dim=0)
        self.log_softmax_d0 = nn.LogSoftmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.student_device = student_device
        self.teacher_device = teacher_device

        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        # adding this conv 1x1 blocks to increase the # of features in student's feat map
        # so that the student and teacher will have same number of channesl
        # which enable us to take the diff of these feature maps
        self.channel_change_conv = nn.ModuleList()
        for sc, tc in zip(self.student_channels, self.teacher_channels):
            if sc != tc:
                from .network_blocks import BaseConv
                conv = BaseConv(sc, tc, 1, 1, bias=False, act='silu').to(self.student_device)
            else:
                conv = nn.Identity().to(self.student_device)
            self.channel_change_conv.append(conv)

        self.grids = [torch.zeros(1)] * len(in_channels)
        self.n_anchors = 1
        self.num_classes = num_classes
        self.strides = strides


    def forward(self, student_feat_map, teacher_feat_map, \
                    student_fpn_feat, teacher_fpn_feat, \
                    labels):
        # student_feat_map : a dict with key 0, 1, 2 and respective feat maps
        # teacher_feat_map : a dict with key 0, 1, 2 and respective feat maps

        # 0 : [B x 4 x 80 x 80], [B x 1 x 80 x 80], [B x 80 x 80 x 80]
        # 1 : [B x 4 x 40 x 40], [B x 1 x 40 x 40], [B x 80 x 40 x 40]
        # 2 : [B x 4 x 20 x 20], [B x 1 x 20 x 20], [B x 80 x 20 x 20]
        # Note : 2nd index is for classification logits. Sigmoid is not applied on this.

        # student_fpn_feat : a list of 3 feature maps having shapes [1, 128, 80, 80], [1, 256, 40, 40], [1, 512, 20, 20]
        # teacher_fpn_feat : same as above

        # labels : [B, N, 5]    : xywh format + cls-id
        
        loss_pgfi = self.compute_pgfi_loss(student_feat_map, teacher_feat_map, student_fpn_feat, teacher_fpn_feat)
        loss_rm = self.compute_rm_loss_iou(student_feat_map, teacher_feat_map, labels)

        return loss_rm, loss_pgfi


    def __compute_iou(self, bb1, bb2):
        # bb1 : [no of boxes in label, 1, 4] 
        # bb2 : [1, n_anchors, 4]

        n_boxes_gt = bb1.shape[0]
        n_anchor_loc = bb2.shape[1]

        bb1 = torch.tile(bb1, (1, n_anchor_loc, 1))
        bb2 = torch.tile(bb2, (n_boxes_gt, 1, 1))

        assert bb1.shape == bb2.shape

        tl = torch.max(
            (bb1[:, :, :2] - bb1[:, :, 2:] / 2), (bb2[:, :, :2] - bb2[:, :, 2:] / 2)
        )
        br = torch.min(
            (bb1[:, :, :2] + bb1[:, :, 2:] / 2), (bb2[:, :, :2] + bb2[:, :, 2:] / 2)
        )

        area_bb1 = torch.prod(bb1[:, :, 2:], 2)
        area_bb2 = torch.prod(bb2[:, :, 2:], 2)

        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        area_u = area_bb1 + area_bb2 - area_i
        iou = (area_i) / (area_u + 1e-16)
        # [no of boxes in label, n_anchors]
        return iou


    def compute_rm_loss_iou(self, student_feat_map, teacher_feat_map, labels):
        loss_rm = 0

        st_outputs = []
        te_outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for (k, (st_reg_output, st_obj_output, st_cls_output)), \
            (_, (te_reg_output, te_obj_output, te_cls_output)) in \
                zip(student_feat_map.items(), teacher_feat_map.items()):    
            stride_this_level = self.strides[k]
            
            st_output = torch.cat([st_reg_output, st_obj_output, st_cls_output], 1)
            te_output = torch.cat([te_reg_output, te_obj_output, te_cls_output], 1)
            
            st_output, grid = self.get_output_and_grid(
                st_output, k, stride_this_level, st_reg_output[0].type()
            )
            te_output, _ = self.get_output_and_grid(
                te_output, k, stride_this_level, te_reg_output[0].type()
            )

            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type_as(st_reg_output[0])
            )

            st_outputs.append(st_output)
            te_outputs.append(te_output)

        # n_anchors_all = total_prediction_across_scales
        st_outputs = torch.cat(st_outputs, 1) # [batch, total_prediction_across_scales, 85]
        te_outputs = torch.cat(te_outputs, 1) # [batch, total_prediction_across_scales, 85]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        
        st_bbox_preds = st_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        # st_cls_preds = st_outputs[:, :, 5:]   # [batch, n_anchors_all, n_cls]
        
        te_bbox_preds = te_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        # te_cls_preds = te_outputs[:, :, 5:]   # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = st_bbox_preds.shape[1]

        num_gts = 0
        batch_size = st_outputs.shape[0]
        for batch_idx in range(batch_size):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                continue
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]       # [no of boxes in label, 4] in xywh manner
                
                st_bbox_preds_per_image = st_bbox_preds[batch_idx]                  # [n_anchors, 4] in xywh manner
                te_bbox_preds_per_image = te_bbox_preds[batch_idx]                  # [n_anchors, 4] in xywh manner


            # Now we will find iou value for each gt box at each location
            st_bbox_preds_per_image_expand = torch.unsqueeze(st_bbox_preds_per_image, dim=0)   # [1, n_anchors, 4]
            te_bbox_preds_per_image_expand = torch.unsqueeze(te_bbox_preds_per_image, dim=0)   # [1, n_anchors, 4]
            gt_bboxes_per_image_expand = torch.unsqueeze(gt_bboxes_per_image, dim=1)  # [no of boxes in label, 1, 4]

            st_iou_map = self.__compute_iou(gt_bboxes_per_image_expand, st_bbox_preds_per_image_expand)        # [no of boxes in label, n_anchors]
            te_iou_map = self.__compute_iou(gt_bboxes_per_image_expand, te_bbox_preds_per_image_expand)        # [no of boxes in label, n_anchors]

            # expanded_strides : [1, total_anchors_across_scales]
            # x_shifts : [1, total_anchors_across_scales]
            # y_shifts : [1, total_anchors_across_scales]

            x_shifts_per_image = x_shifts * expanded_strides
            y_shifts_per_image = y_shifts * expanded_strides
            x_centers_per_image = (
                (x_shifts_per_image + 0.5 * expanded_strides)
                .repeat(num_gt, 1)
            )  # [n_anchor] -> [n_gt, n_anchor]
            y_centers_per_image = (
                (y_shifts_per_image + 0.5 * expanded_strides)
                .repeat(num_gt, 1)
            )

            gt_bboxes_per_image_l = (
                (gt_bboxes_per_image[:, 0:1] - 0.5 * gt_bboxes_per_image[:, 2:3])
                .repeat(1, total_num_anchors)
            )   # [no of boxes in label] -> [no of boxes in label, 4]
            gt_bboxes_per_image_r = (
                (gt_bboxes_per_image[:, 0:1] + 0.5 * gt_bboxes_per_image[:, 2:3])
                .repeat(1, total_num_anchors)
            )
            gt_bboxes_per_image_t = (
                (gt_bboxes_per_image[:, 1:2] - 0.5 * gt_bboxes_per_image[:, 3:4])
                .repeat(1, total_num_anchors)
            )
            gt_bboxes_per_image_b = (
                (gt_bboxes_per_image[:, 1:2] + 0.5 * gt_bboxes_per_image[:, 3:4])
                .repeat(1, total_num_anchors)
            )

            b_l = x_centers_per_image - gt_bboxes_per_image_l   # [no of boxes in label, n_anchors]
            b_r = gt_bboxes_per_image_r - x_centers_per_image   # [no of boxes in label, n_anchors]
            b_t = y_centers_per_image - gt_bboxes_per_image_t   # [no of boxes in label, n_anchors]
            b_b = gt_bboxes_per_image_b - y_centers_per_image   # [no of boxes in label, n_anchors]
            bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # [no of boxes in label, n_anchors, 4]

            is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0      
            # This will give us the mask for each gt, 1 for box present and 0 for box absent at that grid point.
            # [no of boxes in label, n_anchors]
            is_in_boxes_all = is_in_boxes.sum(dim=0) > 0            
            # This will give us the mask where 1 represents the any gt box is present and 0 represents no box at that grid point.
            # [n_anchors]


            rm_loss_per_image = 0
            for label_id in range(num_gt):
                label_mask = is_in_boxes[label_id]                      # [n_anchors]

                st_iou_inside_gt_boxes = st_iou_map[label_id][label_mask]       # [n]   : n points inside gt boxes and each location will have iou value with that gt
                # e.g. if 100 points are inside the current gt box then for those 100 points, we computed iou with label using predicted student and teacher coordinates.
                te_iou_inside_gt_boxes = te_iou_map[label_id][label_mask]       # [n]   : n points inside gt boxes and each location will have iou value with that gt

                st_iou_inside_gt_boxes_ls = self.log_softmax_d0(st_iou_inside_gt_boxes)
                # Then apply softmax over all the softmaxed iou values of those 100 points 
                # st_iou_inside_gt_boxes = self.softmax_d0(st_iou_inside_gt_boxes)
                te_iou_inside_gt_boxes = self.softmax_d0(te_iou_inside_gt_boxes)

                rm_loss_kl_div_per_gt = torch.nn.KLDivLoss(reduction='sum')(st_iou_inside_gt_boxes_ls, te_iou_inside_gt_boxes)
                rm_loss_per_image += rm_loss_kl_div_per_gt

            rm_loss_per_image = rm_loss_per_image / num_gt
            loss_rm += rm_loss_per_image

        loss_rm = loss_rm / batch_size
        return loss_rm



    def compute_rm_loss_cls_prob(self, student_feat_map, teacher_feat_map, labels):
        loss_rm = 0

        st_outputs = []
        te_outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for (k, (st_reg_output, st_obj_output, st_cls_output)), \
            (_, (te_reg_output, te_obj_output, te_cls_output)) in \
                zip(student_feat_map.items(), teacher_feat_map.items()):    
            stride_this_level = self.strides[k]
            
            st_output = torch.cat([st_reg_output, st_obj_output, st_cls_output], 1)
            te_output = torch.cat([te_reg_output, te_obj_output, te_cls_output], 1)
            
            st_output, grid = self.get_output_and_grid(
                st_output, k, stride_this_level, st_reg_output[0].type()
            )
            te_output, _ = self.get_output_and_grid(
                te_output, k, stride_this_level, te_reg_output[0].type()
            )

            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type_as(st_reg_output[0])
            )

            st_outputs.append(st_output)
            te_outputs.append(te_output)

        # n_anchors_all = total_prediction_across_scales
        st_outputs = torch.cat(st_outputs, 1) # [batch, total_prediction_across_scales, 85]
        te_outputs = torch.cat(te_outputs, 1) # [batch, total_prediction_across_scales, 85]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        
        st_bbox_preds = st_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        st_cls_preds = st_outputs[:, :, 5:]   # [batch, n_anchors_all, n_cls]
        
        te_bbox_preds = te_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        te_cls_preds = te_outputs[:, :, 5:]   # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = st_outputs.shape[1]

        num_gts = 0
        batch_size = st_outputs.shape[0]
        for batch_idx in range(batch_size):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                continue
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]       # [no of boxes in label, 4] in xywh manner
                gt_classes = labels[batch_idx, :num_gt, 0]                  # [no of boxes in label]
                st_cls_preds_per_image = st_cls_preds[batch_idx]                  # [n_anchors, n_cls]
                te_cls_preds_per_image = te_cls_preds[batch_idx]                  # [n_anchors, n_cls]

            # expanded_strides : [1, total_anchors_across_scales]
            # x_shifts : [1, total_anchors_across_scales]
            # y_shifts : [1, total_anchors_across_scales]

            x_shifts_per_image = x_shifts * expanded_strides
            y_shifts_per_image = y_shifts * expanded_strides
            x_centers_per_image = (
                (x_shifts_per_image + 0.5 * expanded_strides)
                .repeat(num_gt, 1)
            )  # [n_anchor] -> [n_gt, n_anchor]
            y_centers_per_image = (
                (y_shifts_per_image + 0.5 * expanded_strides)
                .repeat(num_gt, 1)
            )

            gt_bboxes_per_image_l = (
                (gt_bboxes_per_image[:, 0:1] - 0.5 * gt_bboxes_per_image[:, 2:3])
                .repeat(1, total_num_anchors)
            )   # [no of boxes in label] -> [no of boxes in label, 4]
            gt_bboxes_per_image_r = (
                (gt_bboxes_per_image[:, 0:1] + 0.5 * gt_bboxes_per_image[:, 2:3])
                .repeat(1, total_num_anchors)
            )
            gt_bboxes_per_image_t = (
                (gt_bboxes_per_image[:, 1:2] - 0.5 * gt_bboxes_per_image[:, 3:4])
                .repeat(1, total_num_anchors)
            )
            gt_bboxes_per_image_b = (
                (gt_bboxes_per_image[:, 1:2] + 0.5 * gt_bboxes_per_image[:, 3:4])
                .repeat(1, total_num_anchors)
            )

            b_l = x_centers_per_image - gt_bboxes_per_image_l   # [no of boxes in label, n_anchors]
            b_r = gt_bboxes_per_image_r - x_centers_per_image   # [no of boxes in label, n_anchors]
            b_t = y_centers_per_image - gt_bboxes_per_image_t   # [no of boxes in label, n_anchors]
            b_b = gt_bboxes_per_image_b - y_centers_per_image   # [no of boxes in label, n_anchors]
            bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # [no of boxes in label, n_anchors, 4]

            is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0      
            # This will give us the mask for each gt, 1 for box present and 0 for box absent at that grid point.
            # [no of boxes in label, n_anchors]
            is_in_boxes_all = is_in_boxes.sum(dim=0) > 0            
            # This will give us the mask where 1 represents the any gt box is present and 0 represents no box at that grid point.
            # [n_anchors]

            rm_loss_per_image = 0
            for label_id in range(num_gt):
                label_mask = is_in_boxes[label_id]                      # [n_anchors]
                gt_cls_idx = gt_classes[label_id].to(torch.long)        # []

                st_cls_preds_inside_boxes = st_cls_preds_per_image[label_mask][:, gt_cls_idx]      # [gt class predicted probability at no. of points which are associated with the current gt label]
                # e.g. if 100 points are inside this gt box and gt label is "person"
                # then those 100 points' "person" class probability is given here.
                
                te_cls_preds_inside_boxes = te_cls_preds_per_image[label_mask][:, gt_cls_idx]      # [gt class predicted probability at no. of points which are associated with the current gt label]


                # As these are un-sigmoided scores, we first apply sigmoid.
                st_cls_preds_inside_boxes = self.sigmoid(st_cls_preds_inside_boxes)
                # Then apply softmax over all the values of those "person" class to rank them.
                st_cls_preds_inside_boxes = self.softmax_d0(st_cls_preds_inside_boxes)

                te_cls_preds_inside_boxes = self.sigmoid(te_cls_preds_inside_boxes)
                te_cls_preds_inside_boxes = self.softmax_d0(te_cls_preds_inside_boxes)

                rm_loss_kl_div_per_gt = te_cls_preds_inside_boxes * torch.log(st_cls_preds_inside_boxes / (te_cls_preds_inside_boxes + 1e-10))
                rm_loss_kl_div_per_gt = -torch.sum(rm_loss_kl_div_per_gt)
                rm_loss_per_image += rm_loss_kl_div_per_gt

            rm_loss_per_image = rm_loss_per_image / num_gt
            loss_rm += rm_loss_per_image

        loss_rm = loss_rm / batch_size
        return loss_rm


    def compute_pgfi_loss(self, student_feat_map, teacher_feat_map, student_fpn_feat, teacher_fpn_feat):
        
        loss_pgfi = 0
        total_fpn_maps = len(student_fpn_feat)
        for i in range(total_fpn_maps):
            assert teacher_fpn_feat[i].shape[2:4] == student_fpn_feat[i].shape[2:4], "Feature map Height and Width must match for student and teacher network."

            assert teacher_fpn_feat[i].shape[1] == self.teacher_channels[i], "Teacher channels are not provided as per the architecture."
            assert student_fpn_feat[i].shape[1] == self.student_channels[i], "Student channels are not provided as per the architecture."

            student_fpn_feat_mod = self.channel_change_conv[i](student_fpn_feat[i])
            feat_diff = torch.mean(self.mse_no_red(teacher_fpn_feat[i], student_fpn_feat_mod), dim=1)
            # take a mean across feature axis: [1, 128, 80, 80] --> [1, 80, 80]

            pred_diff = torch.mean(self.mse_no_red(self.softmax_d1(teacher_feat_map[i][2]), self.softmax_d1(student_feat_map[i][2])), dim=1)
            # at index 2 we have class prediction tensor and these are logits (no softmax is applied)
            # take a mean across class axis: [1, 80, 80, 80] --> [1, 80, 80]

            feat_h, feat_w = feat_diff.shape[1:]
            assert feat_h == feat_w
            masked_diff = torch.square(torch.linalg.norm(feat_diff * pred_diff, keepdim=True, dim=[1, 2]))
            masked_diff = masked_diff / (feat_h * feat_w)
            masked_diff = torch.mean(masked_diff)
            loss_pgfi += masked_diff
        
        loss_pgfi = loss_pgfi / total_fpn_maps
        return loss_pgfi


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


