from .detector3d_template import Detector3DTemplate
from ...utils import loss_utils, common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn.modules.batchnorm import _BatchNorm


class PointRCNNMeanTeacherMerge(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.smoothl1 = loss_utils.WeightedSmoothL1Loss()

    def set_momemtum_value_for_bn(self, momemtum=0.1):
        def apply_fn(m, momemtum=momemtum):
            if isinstance(m, _BatchNorm):
                m.momentum = momemtum
        self.apply(apply_fn)

    def reset_bn_stats(self):
        def apply_fn(m):
            if isinstance(m, _BatchNorm):
                m.reset_running_stats()
        self.apply(apply_fn)
        
    
    def split_batch_dicts(self, batch_dict):
        # split source and target dict for proposal generation and target assignments
        batch_source, batch_target = {}, {}
        for key, val in batch_dict.items():
            if key == 'batch_size':
                batch_size = val // 2
                batch_source[key], batch_target[key] = batch_size, batch_size
            elif key in ['batch_type', 'cls_preds_normalized', 'has_class_labels']:
                batch_source[key], batch_target[key] = val, val
            elif key == 'gt_boxes':
                batch_source[key] = val
            else:
                split_length = val.shape[0] // 2
                batch_source[key], batch_target[key] = val[:split_length], val[split_length:]
        
        batch_target['points'][:,0] -= batch_target['batch_size']
        
        return batch_source, batch_target

    def forward(self, batch_dict, is_ema=False, cur_epoch=None, ema_model=None):
        if is_ema:
            if isinstance(batch_dict, dict):
                for cur_module in self.module_list:
                    if cur_module.__class__.__name__ not in ['PointRCNNHeadMT', 'PointRCNNHeadMTMerge']:
                        batch_dict = cur_module(batch_dict)
                    else:
                        batch_dict = cur_module(batch_dict)

                return batch_dict
            else:
                raise NotImplementedError
        
        else:
            assert isinstance(batch_dict, list)
            assert len(batch_dict) == 2
            batch_merge, batch_target2 = batch_dict

            # forward source and target together
            for cur_module in self.module_list:
                if cur_module.__class__.__name__ not in ['PointRCNNHeadMT', 'PointRCNNHeadMTMerge']:
                    batch_merge = cur_module(batch_merge)
                else:
                    is_warm_up = self.model_cfg.WARM_UP_EPOCH > cur_epoch
                    if is_warm_up:
                        batch_merge = cur_module(batch_merge)
                    else:
                        batch_merge = cur_module([batch_merge, batch_target2])

            # split source and target batch
            _, batch_target1 = self.split_batch_dicts(batch_merge)

            # do post processing for target data
            pred_dicts1, _ = self.post_processing(batch_target1)
            pred_dicts2, _ = self.post_processing(batch_target2)

            assert self.training

            loss1, tb_dict1, disp_dict1 = self.get_training_loss()

            if self.model_cfg.get('WARM_UP_EPOCH', 0) > 0 and cur_epoch < self.model_cfg.WARM_UP_EPOCH:
                loss2, tb_dict2 = 0, {}
                loss3, tb_dict3 = 0, {}
            else:
                if self.model_cfg.CONSISTENCY_LOSS['roi_cls_weight'] + self.model_cfg.CONSISTENCY_LOSS['roi_reg_weight'] > 0:
                    loss2, tb_dict2 = self.get_consistency_loss_roi(batch_target1, batch_target2, pred_dicts1, pred_dicts2)
                else:
                    loss2, tb_dict2 = 0, {}
                
                if self.model_cfg.CONSISTENCY_LOSS['rcnn_cls_weight'] + self.model_cfg.CONSISTENCY_LOSS['rcnn_reg_weight']:
                    loss3, tb_dict3 = self.get_consistency_loss_rcnn(batch_target1, batch_target2)
                else:
                    loss3, tb_dict3 = 0, {}
            
            tb_dict1.update(tb_dict2)
            tb_dict1.update(tb_dict3)

            if self.model_cfg.get('WARM_UP_EPOCH', 0) > 0 and cur_epoch < self.model_cfg.WARM_UP_EPOCH:
                source_loss_weight = self.model_cfg.get('SOURCE_LOSS_WEIGHT', 1.0)
                loss = loss1 * source_loss_weight
            else:
                source_loss_weight = self.model_cfg.get('SOURCE_LOSS_WEIGHT', 1.0)
                if self.model_cfg.get('SOURCE_LOSS_SCHEDULE', 'off') == 'exp':
                    multiplier = np.exp(-cur_epoch)
                else:
                    multiplier = 1.0
                loss = loss1 * source_loss_weight * multiplier + loss2 + loss3
                
            ret_dict = {
                'loss': loss
            }
            disp_dict1.update(tb_dict2)
            disp_dict1.update(tb_dict3)

            return ret_dict, tb_dict1, disp_dict1


    def get_consistency_loss_rcnn(self, batch_target1, batch_target2, return_reg_target=False):
        cls_pred1 = batch_target1['batch_cls_preds']
        box_pred1 = batch_target1['batch_box_preds']
        cls_pred2 = batch_target2['batch_cls_preds']
        box_pred2 = batch_target2['batch_box_preds']
        rois1 = batch_target1['rois']
        rois2 = batch_target2['rois']
        code_size = box_pred2.shape[-1]
        batch_size = batch_target1['batch_size']
        roi_size = batch_target1['rois'].shape[1]
        
        box_pred2 = common_utils.reverse_augmentation(box_pred2, batch_target2)
        box_pred2 = common_utils.forward_augmentation(box_pred2, batch_target1)
        rois2 = common_utils.reverse_augmentation(rois2, batch_target2)
        rois2 = common_utils.forward_augmentation(rois2, batch_target1)

        batch_target1['prediction_from_ema'] = box_pred2.clone().detach()

        match_cls_th = self.model_cfg.get('RCNN_FILTER_IOU_THRESHOLD', 0.99)
        assert self.model_cfg.ROI_HEAD.LOSS_CONFIG.CLS_LOSS == 'BinaryCrossEntropy'
        matching_mask1 = (torch.sigmoid(cls_pred1.squeeze()) > match_cls_th).float()
        matching_mask2 = (torch.sigmoid(cls_pred2.squeeze()) > match_cls_th).float()
        matching_mask = matching_mask1 * matching_mask2

        if self.model_cfg.RCNN_FILTER_BY_POSITIVE_ROIS > 0:
            matching_mask_sum = matching_mask.sum(-1)
            matching_mask *= (matching_mask_sum >= self.model_cfg.RCNN_FILTER_BY_POSITIVE_ROIS).float().unsqueeze(-1)

        # cls consistency loss
        prob1 = torch.sigmoid(cls_pred1).view(batch_size*roi_size, 1)
        prob2 = torch.sigmoid(cls_pred2).view(batch_size*roi_size, 1)
        prob1 = prob1.clamp(1e-4, 1-1e-4)
        prob2 = prob2.clamp(1e-4, 1-1e-4)
        prob1 = torch.cat([1-prob1, prob1], dim=-1)
        prob2 = torch.cat([1-prob2, prob2], dim=-1)
        cls_consistency_loss = F.kl_div(torch.log(prob1), prob2)

        # if self.model_cfg.ENCODED_RCNN_CONSISTENCY:
        rcnn_reg = batch_target1['rcnn_reg']
        # rcnn_cls = batch_target1['rcnn_cls']
        rois = batch_target1['rois'].clone().detach()
        gt_of_rois = box_pred2.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry
        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)
        gt_of_rois[:, :, 6] = heading_label
        # encode
        code_size = self.roi_head.box_coder.code_size
        gt_boxes3d_ct = gt_of_rois[..., 0:code_size]
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]
        rois_anchor = rois.clone().detach().view(-1, code_size)
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        reg_targets = self.roi_head.box_coder.encode_torch(
            gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
        )

        rcnn_loss_reg = self.roi_head.reg_loss_func(
            rcnn_reg.unsqueeze(dim=0),
            reg_targets.unsqueeze(dim=0),
        )  # [B, M, 7] 

        fg_mask = matching_mask

        reg_consistency_loss = (rcnn_loss_reg * fg_mask.view(rcnn_batch_size, -1)).sum() / max(fg_mask.sum(), 1)

        loss = 0
        if self.model_cfg.CONSISTENCY_LOSS['rcnn_cls_weight'] > 0:
            loss += cls_consistency_loss 
        if self.model_cfg.CONSISTENCY_LOSS['rcnn_reg_weight'] > 0:
            loss += reg_consistency_loss 

        tb_dict = {'rcnn_cls_cons': cls_consistency_loss.item(), 
                   'rcnn_reg_cons': reg_consistency_loss.item()
                   }

        return loss, tb_dict


    def get_consistency_loss_roi(self, batch_target1, batch_target2, pred_dicts1, pred_dicts2):
        
        fg_mask = self.get_fg_mask(batch_target1, batch_target2, pred_dicts1, pred_dicts2)
        if fg_mask.sum() == 0:
            return 0, {}

        cls_pred1 = batch_target1['batch_cls_preds_roi']
        box_pred1 = batch_target1['batch_box_preds_roi']
        cls_pred2 = batch_target2['batch_cls_preds_roi']
        box_pred2 = batch_target2['batch_box_preds_roi']
        batch_size = batch_target2['batch_size']
        code_size = box_pred2.shape[1]
        box_pred2 = box_pred2.view(batch_size, -1, code_size)

        box_pred2 = common_utils.reverse_augmentation(box_pred2, batch_target2)
        box_pred2 = common_utils.forward_augmentation(box_pred2, batch_target1)
        box_pred2 = box_pred2.view(-1, code_size)

        prob1 = torch.sigmoid(cls_pred1) # (N, n_class)
        prob2 = torch.sigmoid(cls_pred2)
        prob1 = prob1.clamp(1e-4, 1-1e-4)
        prob2 = prob2.clamp(1e-4, 1-1e-4)
        if cls_pred1.shape[-1] == 1:
            prob1 = torch.cat([1-prob1, prob1], dim=1)
            prob2 = torch.cat([1-prob2, prob2], dim=1)
        else: # multiple classes
            prob1 = torch.stack([1-prob1, prob1], dim=-1)
            prob2 = torch.stack([1-prob2, prob2], dim=-1)
        cls_consistency_loss = F.kl_div(torch.log(prob1), prob2, reduction='none')
        cls_consistency_loss = cls_consistency_loss.mean()

        selected_box_pred1 = box_pred1[fg_mask]
        selected_box_pred2 = box_pred2[fg_mask].clone().detach()

        # save selected_boxes
        batch_target1['selected_box_pred'] = selected_box_pred1
        batch_target2['selected_box_pred'] = selected_box_pred2
        
        points = batch_target1['points']
        reg_preds = batch_target1['point_box_preds_encoded']
        reg_labels = points.new_zeros((points.shape[0], 8))
        fg_reg_labels = self.point_head.box_coder.encode_torch(gt_boxes=selected_box_pred2, 
                                    points=points[fg_mask][...,1:4], gt_classes=selected_box_pred2.new_ones((fg_mask.sum())).long())
        reg_labels[fg_mask] = fg_reg_labels

        # reg loss computation
        reg_weights = fg_mask.float()
        pos_normalizer = fg_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_consistency_loss = self.smoothl1(reg_preds[None, ...], reg_labels[None, ...], weights=reg_weights[None, ...]).sum()

        cls_consistency_loss *= self.model_cfg.CONSISTENCY_LOSS['roi_cls_weight']
        reg_consistency_loss *= self.model_cfg.CONSISTENCY_LOSS['roi_reg_weight']

        loss = cls_consistency_loss + reg_consistency_loss

        tb_dict = {'roi_cls_consist': cls_consistency_loss.item(), 
                   'roi_reg_consist': reg_consistency_loss.item(),
                   'fg_points': (fg_mask.sum()/batch_size).item()}

        return loss, tb_dict

    def get_fg_mask(self, batch_target1, batch_target2, pred_dicts1, pred_dicts2):
        pred_boxes1 = [x['pred_boxes'] for x in pred_dicts1] # list of pred boxes 
        pred_boxes2 = [x['pred_boxes'] for x in pred_dicts2] # list of pred boxes 
        points = batch_target1['points'] # （N, 4)

        batch_size = len(pred_boxes1)
        code_size = pred_boxes1[0].shape[1]
        num_roi = batch_target1['rois'].shape[1]
        boxes1 = pred_boxes1[0].new_zeros((batch_size, num_roi, code_size))
        boxes2 = pred_boxes1[0].new_zeros((batch_size, num_roi, code_size))
        for index in range(batch_size):
            pred1 = pred_boxes1[index]
            pred2 = pred_boxes2[index]
            boxes1[index, :pred1.shape[0]] = pred1
            boxes2[index, :pred2.shape[0]] = pred2
        boxes2 = common_utils.reverse_augmentation(boxes2, batch_target2)
        boxes2 = common_utils.forward_augmentation(boxes2, batch_target1)

        bs_idx = points[:, 0]
        fg_masks = []
        
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            box_idxs_of_pts1 = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), boxes1[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
            box_idxs_of_pts2 = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), boxes2[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
            fg_flag = (box_idxs_of_pts1 >= 0) & (box_idxs_of_pts2 >= 0)
            fg_masks.append(fg_flag)

        fg_mask = torch.cat(fg_masks, dim=0)
        batch_target1['fg_mask'] = fg_mask
        batch_target1['pred_boxes'] = pred_boxes1
        batch_target2['pred_boxes'] = pred_boxes2

        return fg_mask

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        disp_dict['loss_point'] = loss_point.item()
        disp_dict['loss_rcnn'] = loss_rcnn.item()
        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

