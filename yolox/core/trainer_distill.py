#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, student_exp, teacher_exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.student_exp = student_exp
        self.teacher_exp = teacher_exp
        self.args = args

        # training related attr
        self.max_epoch = student_exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.student_device = "cuda:{}".format(self.local_rank)
        # self.teacher_device = "cpu"
        self.teacher_device = self.student_device
        self.use_model_ema = student_exp.ema
        self.save_history_ckpt = student_exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.student_input_size = student_exp.input_size
        self.teacher_input_size = teacher_exp.input_size
        assert self.student_input_size == self.teacher_input_size, "The input resolution of student and teacher should be same so as to match the final feature maps and compare the values in loss."
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=student_exp.print_interval)
        self.file_name = os.path.join(student_exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

        self.yolox_loss = self.student_exp.get_yolox_loss()
        
        if self.student_exp.kd_loss_type == 'NORMAL':
            from yolox.models import KDLoss
            self.kd_loss = KDLoss(
                                temperature=self.student_exp.temperature, \
                                kd_cls_weight=self.student_exp.kd_cls_weight, \
                                kd_hint_weight=self.student_exp.kd_hint_weight, \
                                pos_cls_weight=self.student_exp.pos_cls_weight if self.student_exp.has_background_class else None, \
                                neg_cls_weight=self.student_exp.neg_cls_weight if self.student_exp.has_background_class else None, \
                                student_device=self.student_device, \
                                teacher_device=self.teacher_device)
        elif self.student_exp.kd_loss_type == 'RM_PGFI':
            from yolox.models import KDLoss_RM_PGFI
            self.kd_loss = KDLoss_RM_PGFI(
                                student_channels=[int(s * self.student_exp.width) for s in self.student_exp.in_channels], \
                                teacher_channels=[int(s * self.teacher_exp.width) for s in self.teacher_exp.in_channels], \
                                student_device=self.student_device, \
                                teacher_device=self.teacher_device, \
                                in_channels=self.student_exp.in_channels, \
                                num_classes=self.student_exp.num_classes)


    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        # print(self.data_type)
        # exit()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.student_exp.preprocess(inps, targets, self.student_input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            if self.student_exp.use_fpn_feats:
                student_output_fmaps, student_fpn_fmaps = self.student_model(inps, targets)
            else:
                student_output_fmaps = self.student_model(inps, targets)
            old_val = self.teacher_model.head.get_fmaps_only
            self.teacher_model.head.get_fmaps_only = True
            if self.student_exp.use_fpn_feats:
                teacher_output_fmaps, teacher_fpn_fmaps = self.teacher_model(inps, targets)
            else:
                teacher_output_fmaps = self.teacher_model(inps, targets)
            self.teacher_model.head.get_fmaps_only = old_val

        # for s, t in zip(student_output_fmaps, teacher_output_fmaps):
        #     print(s.shape)
        #     print(t.shape)
        #     print("==="*30)

            loss_iou, loss_obj, loss_cls, loss_l1, num_fg = self.yolox_loss(student_output_fmaps, targets)

            if self.student_exp.use_fpn_feats:
                self.old_copy = list(self.kd_loss.parameters())[0].data

                self.kd_loss.train()
                loss_rm, loss_pgfi = self.kd_loss(student_output_fmaps, teacher_output_fmaps, \
                                                    student_fpn_fmaps, teacher_fpn_fmaps, targets)
                loss_rm = loss_rm * self.student_exp.rm_alpha
                loss_pgfi = loss_pgfi * self.student_exp.pgfi_beta
                
                loss_cls_kd = 0
                loss_kd_hint = 0
            else:
                loss_kd_softmax_temp, loss_kd_hint = self.kd_loss(student_output_fmaps, teacher_output_fmaps)

                loss_cls = (1 - self.student_exp.kd_cls_weight) * loss_cls
                loss_cls_kd = self.student_exp.kd_cls_weight * loss_kd_softmax_temp

                loss_rm = 0
                loss_pgfi = 0

            loss_total = loss_iou + loss_obj + loss_cls + loss_cls_kd + loss_l1 + loss_kd_hint + loss_rm + loss_pgfi

        outputs = {
            "total_loss": loss_total,
            "iou_loss": loss_iou,
            "l1_loss": loss_l1,
            "conf_loss": loss_obj,
            "cls_loss": loss_cls,
            "num_fg": num_fg,
        }

        if self.student_exp.use_fpn_feats:
            outputs['kd_loss_rm'] = loss_rm
            outputs['kd_loss_pgfi'] = loss_pgfi
        else:
            outputs['kd_cls_loss'] = loss_cls_kd
            outputs['kd_hint_loss'] = loss_kd_hint
        
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.student_model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("Student exp value:\n{}".format(self.student_exp))
        logger.info("Teacher exp value:\n{}".format(self.teacher_exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        student_model = self.student_exp.get_model()
        teacher_model = self.teacher_exp.get_model()
        logger.info(
            "Student Model Summary: {}".format(
                get_model_info(student_model, self.student_exp.test_size))
        )
        logger.info(
            "Teacher Model Summary: {}".format(
                get_model_info(teacher_model, self.teacher_exp.test_size))
        )
        student_model.to(self.student_device)
        teacher_model.to(self.teacher_device)

        # solver related init
        self.optimizer = self.student_exp.get_optimizer(self.args.batch_size)

        if self.student_exp.kd_loss_type == 'RM_PGFI':
            import torch.nn as nn
            pg_rm_pgfi = []
            for k, v in self.kd_loss.named_modules():
                if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg_rm_pgfi.append(v.weight)  # apply decay
            self.optimizer.add_param_group({"params": pg_rm_pgfi})

        # Restore teacher checkpoint
        teacher_model = self.load_teacher(teacher_model)

        # value of epoch will be set in `resume_train`
        # Call this method only for student model only
        student_model = self.resume_student_train(student_model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.student_exp.no_aug_epochs
        self.train_loader = self.student_exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.student_exp.get_lr_scheduler(
            self.student_exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        assert not self.is_distributed, "Distributed training is not supported."
        # if self.is_distributed:
        #     model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(student_model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.student_model = student_model
        self.student_model.train()
        self.teacher_model = teacher_model
        self.teacher_model.eval()       # We dont want to train teacher model

        self.evaluator = self.student_exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))

        logger.info("Training start...")
        # logger.info("\n{}".format(student_model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.student_exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                # self.student_model.module.head.use_l1 = True
                self.yolox_loss.use_l1 = True
            else:
                # self.student_model.head.use_l1 = True
                self.yolox_loss.use_l1 = True
            self.student_exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.student_exp.eval_interval == 0:
            all_reduce_norm(self.student_model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.student_exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.student_input_size[0], eta_str))
            )

            curr_steps = self.epoch * self.max_iter + self.iter + 1
            for k, v in loss_meter.items():
                self.tblogger.add_scalar(k, v.latest, curr_steps)
            self.tblogger.add_scalar("lr", self.meter["lr"].latest, curr_steps)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.student_input_size = self.student_exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter


    def load_teacher(self, model):
        logger.info("loading checkpoint for teacher")
        ckpt_file = self.args.teacher_ckpt
        ckpt = torch.load(ckpt_file, map_location=self.teacher_device)["model"]
        model = load_ckpt(model, ckpt)
        return model

    def resume_student_train(self, model):
        if self.args.resume:
            # Resume the student network training
            logger.info("resume training")
            if self.args.student_ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.student_ckpt

            ckpt = torch.load(ckpt_file, map_location=self.student_device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            # This should not be called as we want to train student from
            # scratch using knowledge distillation
            self.start_epoch = 0
            return model
            assert False, "Student weights should not be restored if we are not resuming the training."
            if self.args.student_ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.student_model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        evalmodel.return_backbone_feats = False     # Setting this to false will return only final outputs
        ap50_95, ap50, summary = self.student_exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        evalmodel.return_backbone_feats = True      # Need to reset this to True as during training it is required to output other feature maps as well.
        
        # Above function calls model.eval() internally.
        # So to reset that we need to call model.train()
        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        self.student_model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        # self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        # self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", update_best_ckpt)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.student_model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
