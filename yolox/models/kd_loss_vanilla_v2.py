# Created Date: Sunday, January 30th 2022, 9:33:15 am
# Author: meet_minimalist
# Copyright (c) 2022 

from lib2to3.pytree import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class KDLoss_Vanilla_v2(nn.Module):
    def __init__(self, temperature=1.0, kd_cls_weight=0.5, \
                    pos_cls_weight=1.0, neg_cls_weight=1.5, student_device='cuda:0', \
                    teacher_device='cuda:0', in_channels=[256, 512, 1024], num_classes=80, \
                    strides=[8, 16, 32]):
        super(KDLoss_Vanilla_v2, self).__init__()

        self.mse_no_red = nn.MSELoss(reduction='none')
        self.softmax_d1 = nn.Softmax(dim=1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.student_device = student_device
        self.teacher_device = teacher_device

        self.temperature = temperature
        self.kd_cls_weight = kd_cls_weight
        self.pos_w = pos_cls_weight
        self.neg_w = neg_cls_weight
        if self.pos_w is None and self.neg_w is None:
            # Dont use any weighing scheme.
            self.use_background_weight = False

        self.grids = [torch.zeros(1)] * len(in_channels)
        self.n_anchors = 1
        self.num_classes = num_classes
        self.strides = strides


    def weighted_kl_div(self, ps, qt):
        # ps, qt shape : [B, C, H, W]
        eps = 1e-10
        ps = ps + eps
        qt = qt + eps
        log_p = qt * torch.log(ps)
        if self.use_background_weight:
            log_p[:, 0] *= self.neg_w
            log_p[:, 1:] *= self.pos_w

        # return -torch.sum(log_p)
        return torch.mean(-torch.sum(log_p, dim=1))


    def forward(self, student_feat_map, teacher_feat_map):
        # student_feat_map : a dict with key 0, 1, 2 and respective feat maps
        # teacher_feat_map : a dict with key 0, 1, 2 and respective feat maps

        # 0 : [B x 4 x 80 x 80], [B x 1 x 80 x 80], [B x 80 x 80 x 80]
        # 1 : [B x 4 x 40 x 40], [B x 1 x 40 x 40], [B x 80 x 40 x 40]
        # 2 : [B x 4 x 20 x 20], [B x 1 x 20 x 20], [B x 80 x 20 x 20]
        # Note : 2nd index is for classification logits. Sigmoid is not applied on this.

        
        loss_kd_softmax_temp = self.__get_kd_softmax_loss(student_feat_map, teacher_feat_map)
        loss_kd_reg = self.__get_kd_reg_loss(student_feat_map, teacher_feat_map)
        loss_kd_obj = self.__get_kd_obj_loss(student_feat_map, teacher_feat_map)
        return loss_kd_softmax_temp, loss_kd_obj, loss_kd_reg


    def __get_kd_softmax_loss(self, student_feat_map, teacher_feat_map):
        
        loss_kd_softmax_temp = 0
        for (k, (_, _, st_cls_output)), \
            (_, (_, _, te_cls_output)) in \
                zip(student_feat_map.items(), teacher_feat_map.items()):
            # cls : [B x 80 x 20 x 20]
            
            if self.teacher_device != self.student_device:
                te_cls_output = te_cls_output.to(self.student_device)

            # Compute loss for classification task
            conf_st = self.softmax_d1(st_cls_output/self.temperature)
            conf_te = self.softmax_d1(te_cls_output/self.temperature)
            loss_kd_softmax_temp += self.weighted_kl_div(conf_st, conf_te)

        return loss_kd_softmax_temp


    def __get_kd_obj_loss(self, student_feat_map, teacher_feat_map):

        loss_kd_obj = 0
        for (k, (_, st_obj_output, _)), \
            (_, (_, te_obj_output, _)) in \
                zip(student_feat_map.items(), teacher_feat_map.items()):
            # obj : [B x 1 x 80 x 80]
            
            if self.teacher_device != self.student_device:
                te_obj_output = te_obj_output.to(self.student_device)

            # Compute loss for objectness matching
            loss_kd_obj += torch.mean(self.bce_loss(st_obj_output, self.sigmoid(te_obj_output)))
        
        return loss_kd_obj



    def __get_kd_reg_loss(self, student_feat_map, teacher_feat_map):

        loss_kd_reg = 0
        for (k, (st_reg_output, st_obj_output, st_cls_output)), \
            (_, (te_reg_output, te_obj_output, te_cls_output)) in \
                zip(student_feat_map.items(), teacher_feat_map.items()):
            # reg : [B x 4 x 80 x 80] 
            # obj : [B x 1 x 80 x 80]
            
            if self.teacher_device != self.student_device:
                te_reg_output = te_reg_output.to(self.student_device)
                te_obj_output = te_obj_output.to(self.student_device)

            # Compute loss for regression task
            stride_this_level = self.strides[k]
            
            st_output = torch.cat([st_reg_output, st_obj_output, st_cls_output], 1)
            te_output = torch.cat([te_reg_output, te_obj_output, te_cls_output], 1)
            
            st_output, _ = self.get_output_and_grid(
                st_output, k, stride_this_level, st_reg_output[0].type()
            )
            # st_output : [B, num_anchors_per_scale, 85] --> xywh,obj,cls_1,cls_2...
            te_output, _ = self.get_output_and_grid(
                te_output, k, stride_this_level, te_reg_output[0].type()
            )
            # te_output : [B, num_anchors_per_scale, 85] --> xywh,obj,cls_1,cls_2...

            st_x1 = st_output[:, :, 0:1] - st_output[:, :, 2:3] / 2 
            st_y1 = st_output[:, :, 1:2] - st_output[:, :, 3:4] / 2 
            st_x2 = st_output[:, :, 0:1] + st_output[:, :, 2:3] / 2 
            st_y2 = st_output[:, :, 1:2] + st_output[:, :, 3:4] / 2
            st_coords = torch.cat([st_x1, st_y1, st_x2, st_y2], dim=2)
            # [B, num_anchors_per_scale, 4]

            te_x1 = te_output[:, :, 0:1] - te_output[:, :, 2:3] / 2 
            te_y1 = te_output[:, :, 1:2] - te_output[:, :, 3:4] / 2 
            te_x2 = te_output[:, :, 0:1] + te_output[:, :, 2:3] / 2 
            te_y2 = te_output[:, :, 1:2] + te_output[:, :, 3:4] / 2
            te_coords = torch.cat([te_x1, te_y1, te_x2, te_y2], dim=2)
            # [B, num_anchors_per_scale, 4]

            te_obj_score = self.sigmoid(te_output[:, :, 4:5])
            # [B, num_anchors_per_scale, 1]

            # Use teacher's objectness value as a mask to filter the regression loss calculation to only positive samples
            loss_kd_reg_per_scale = te_obj_score * self.mse_no_red(st_coords, te_coords)
            # [B, num_anchors_per_scale, 4]
            loss_kd_reg += torch.mean(torch.sum(loss_kd_reg_per_scale, dim=2))


        return loss_kd_reg
        
        


        # # n_anchors_all = total_prediction_across_scales
        # st_outputs = torch.cat(st_outputs, 1) # [batch, total_prediction_across_scales, 85]
        # te_outputs = torch.cat(te_outputs, 1) # [batch, total_prediction_across_scales, 85]
        # x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        # y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        # expanded_strides = torch.cat(expanded_strides, 1)
        
        # st_bbox_preds = st_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        # # st_cls_preds = st_outputs[:, :, 5:]   # [batch, n_anchors_all, n_cls]
        # st_obj_preds_raw = st_outputs[:, :, 4:5]    # [batch, n_anchors_all, 1]     --> Raw logits for objectness value
        
        # te_bbox_preds = te_outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        # # te_cls_preds = te_outputs[:, :, 5:]   # [batch, n_anchors_all, n_cls]
        # te_obj_preds_raw = te_outputs[:, :, 4:5]    # [batch, n_anchors_all, 1]     --> Raw logits for objectness value

        # # calculate targets
        # nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # total_num_anchors = st_bbox_preds.shape[1]

        # num_gts = 0
        # batch_size = st_outputs.shape[0]
        # for batch_idx in range(batch_size):
        #     num_gt = int(nlabel[batch_idx])
        #     num_gts += num_gt
        #     if num_gt == 0:
        #         continue
        #     else:
        #         gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]       # [no of boxes in label, 4] in xywh manner
                
        #         st_bbox_preds_per_image = st_bbox_preds[batch_idx]                  # [n_anchors, 4] in xywh manner
        #         te_bbox_preds_per_image = te_bbox_preds[batch_idx]                  # [n_anchors, 4] in xywh manner


        #     # Now we will find iou value for each gt box at each location
        #     st_bbox_preds_per_image_expand = torch.unsqueeze(st_bbox_preds_per_image, dim=0)   # [1, n_anchors, 4]
        #     te_bbox_preds_per_image_expand = torch.unsqueeze(te_bbox_preds_per_image, dim=0)   # [1, n_anchors, 4]
        #     gt_bboxes_per_image_expand = torch.unsqueeze(gt_bboxes_per_image, dim=1)  # [no of boxes in label, 1, 4]

        #     st_obj_preds_raw_per_image = st_obj_preds_raw[batch_idx]                    # [n_anchors, 1]
        #     te_obj_preds_raw_per_image = te_obj_preds_raw[batch_idx]                    # [n_anchors, 1]

        #     # expanded_strides : [1, total_anchors_across_scales]
        #     # x_shifts : [1, total_anchors_across_scales]
        #     # y_shifts : [1, total_anchors_across_scales]

        #     x_shifts_per_image = x_shifts * expanded_strides
        #     y_shifts_per_image = y_shifts * expanded_strides
        #     x_centers_per_image = (
        #         (x_shifts_per_image + 0.5 * expanded_strides)
        #         .repeat(num_gt, 1)
        #     )  # [n_anchor] -> [n_gt, n_anchor]
        #     y_centers_per_image = (
        #         (y_shifts_per_image + 0.5 * expanded_strides)
        #         .repeat(num_gt, 1)
        #     )

        #     gt_bboxes_per_image_l = (
        #         (gt_bboxes_per_image[:, 0:1] - 0.5 * gt_bboxes_per_image[:, 2:3])
        #         .repeat(1, total_num_anchors)
        #     )   # [no of boxes in label] -> [no of boxes in label, 4]
        #     gt_bboxes_per_image_r = (
        #         (gt_bboxes_per_image[:, 0:1] + 0.5 * gt_bboxes_per_image[:, 2:3])
        #         .repeat(1, total_num_anchors)
        #     )
        #     gt_bboxes_per_image_t = (
        #         (gt_bboxes_per_image[:, 1:2] - 0.5 * gt_bboxes_per_image[:, 3:4])
        #         .repeat(1, total_num_anchors)
        #     )
        #     gt_bboxes_per_image_b = (
        #         (gt_bboxes_per_image[:, 1:2] + 0.5 * gt_bboxes_per_image[:, 3:4])
        #         .repeat(1, total_num_anchors)
        #     )

        #     b_l = x_centers_per_image - gt_bboxes_per_image_l   # [no of boxes in label, n_anchors]
        #     b_r = gt_bboxes_per_image_r - x_centers_per_image   # [no of boxes in label, n_anchors]
        #     b_t = y_centers_per_image - gt_bboxes_per_image_t   # [no of boxes in label, n_anchors]
        #     b_b = gt_bboxes_per_image_b - y_centers_per_image   # [no of boxes in label, n_anchors]
        #     bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # [no of boxes in label, n_anchors, 4]

        #     is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0      
        #     # This will give us the mask for each gt, 1 for box present and 0 for box absent at that grid point.
        #     # [no of boxes in label, n_anchors]
        #     is_in_boxes_all = is_in_boxes.sum(dim=0) > 0            
        #     # This will give us the mask where 1 represents the any gt box is present and 0 represents no box at that grid point.
        #     # [n_anchors]

        #     kd_reg_loss_per_image = 0
        #     for label_id in range(num_gt):
        #         label_mask = is_in_boxes[label_id]                      # [n_anchors]

        #         st_bbox_preds_per_image_per_gt_inside_box = st_bbox_preds_per_image_expand[0][label_mask]           # [n x 4]
        #         te_bbox_preds_per_image_per_gt_inside_box = te_bbox_preds_per_image_expand[0][label_mask]           # [n x 4]

        #         teacher_obj_mask = self.sigmoid(te_obj_preds_raw_per_image[label_id])   # [n x 1]

        #         weighted_mse = teacher_obj_mask * self.mse_no_red(st_bbox_preds_per_image_per_gt_inside_box, te_bbox_preds_per_image_per_gt_inside_box)
        #         # [n x 4]
                
        #         kd_reg_loss_per_image += torch.sum(weighted_mse)
        #     kd_reg_loss_per_image = kd_reg_loss_per_image / num_gt
        #     loss_kd_reg += kd_reg_loss_per_image

        # loss_kd_reg = loss_kd_reg / batch_size
        # return loss_kd_reg


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


