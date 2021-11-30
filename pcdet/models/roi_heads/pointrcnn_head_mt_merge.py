import torch
import torch.nn as nn
import pdb
import numpy as np
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from ...utils import loss_utils
from .roi_head_template import RoIHeadTemplate


class PointRCNNHeadMTMerge(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]

        if self.model_cfg.LOSS_CONFIG.CLS_LOSS == 'BinaryCrossEntropy':
            self.cls_layers = self.make_fc_layers(
                input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
            )
        elif self.model_cfg.LOSS_CONFIG.CLS_LOSS == 'CrossEntropy':
            self.cls_layers = self.make_fc_layers(
                input_channels=channel_in, output_channels=self.num_class+1, fc_list=self.model_cfg.CLS_FC
            )
        else:
            raise NotImplementedError

        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        
        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
        )

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roipool3d_gpu(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1)
        batch_points = point_coords.view(batch_size, -1, 3)
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_features

    def split_batch_dicts(self, batch_dict):
        # split source and target dict for proposal generation and target assignments
        batch_source, batch_target = {}, {}
        for key, val in batch_dict.items():
            if key == 'batch_size':
                batch_size = val // 2
                batch_source[key], batch_target[key] = batch_size, batch_size
            elif key in ['batch_type', 'cls_preds_normalized']:
                batch_source[key], batch_target[key] = val, val
            elif key == 'gt_boxes':
                batch_source[key] = val
            else:
                split_length = val.shape[0] // 2
                batch_source[key], batch_target[key] = val[:split_length], val[split_length:]
        
        batch_target['batch_index'] -= batch_target['batch_size']

        return batch_source, batch_target

    def merge_batch_dicts(self, batch_source, batch_target):
        batch_dict = {}
        for key, val in batch_source.items():
            if key == 'batch_size':
                batch_dict[key] = val + batch_target[key]
            elif key in ['batch_type', 'cls_preds_normalized', 'has_class_labels']:
                batch_dict[key] = val
            elif key == 'gt_boxes':
                batch_dict[key] = val
            elif key == 'roi_scores': # will cause error due to diff dimension, but actually don't need it
                continue
            else:
                if isinstance(val, np.ndarray):
                    batch_dict[key] = np.concatenate([val, batch_target[key]])
                else:
                    if key == 'point_coords':
                        batch_target[key][:,0] += batch_target['batch_size']
                    batch_dict[key] = torch.cat([val, batch_target[key]], dim=0)

        return batch_dict

    def augment_rois(self, rois):
        if self.model_cfg.ROI_AUGMENTATION:
            roi_scale_range = self.model_cfg.ROI_AUGMENTATION_SCALE_RANGE
            if not (roi_scale_range[0] == roi_scale_range[1] == 1):
                if self.model_cfg.get('SAME_SCALE_XYZ', False):
                    roi_scale_factor = roi_scale_range[0] + torch.rand_like(rois[:,:,3]) * (roi_scale_range[1] - roi_scale_range[0])
                    roi_scale_factor = roi_scale_factor.unsqueeze(-1)
                else:
                    roi_scale_factor = roi_scale_range[0] + torch.rand_like(rois[:,:,3:6]) * (roi_scale_range[1] - roi_scale_range[0])
                rois[...,3:6] *= roi_scale_factor
            roi_rotate_range = self.model_cfg.ROI_AUGMENTATION_ROTATE_RANGE
            if not (roi_rotate_range[0] == roi_rotate_range[1] == 0):
                roi_rotate_angle = roi_rotate_range[0] + torch.rand_like(rois[:,:,6]) * (roi_rotate_range[1] - roi_rotate_range[0])
                rois[...,6] += roi_rotate_angle
        
        return rois

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
        Returns:
        """

        if isinstance(batch_dict, list):
            batch_dict, batch_dict_other = batch_dict
            dual_dict_flag = True
        else:
            dual_dict_flag = False

        if batch_dict['batch_type'] == 'target':
            nms_config = self.model_cfg.NMS_CONFIG['TRAIN_TARGET']
            targets_dict = self.proposal_layer(batch_dict, nms_config=nms_config)

        else:
            assert batch_dict['batch_type'] == 'merge'
            batch_source, batch_target = self.split_batch_dicts(batch_dict)
            targets_dict = self.proposal_layer(batch_source, nms_config=self.model_cfg.NMS_CONFIG['TRAIN'])
            _ = self.proposal_layer(batch_target, nms_config=self.model_cfg.NMS_CONFIG['TRAIN_TARGET']) # don't need target dict for target domain data

            # assign target for source domain
            targets_dict = self.assign_targets(batch_source)
            batch_source['rois'] = targets_dict['rois']
            batch_source['roi_labels'] = targets_dict['roi_labels']
            
        if batch_dict['batch_type'] == 'target':
            assert not dual_dict_flag
            if self.model_cfg.ROI_AUGMENTATION and self.model_cfg.get('AUGMENT_BOTH_STUDENT_AND_TEACHER'):
                batch_dict['rois_before_aug'] = batch_dict['rois'].clone().detach()
                batch_dict['rois'] = self.augment_rois(batch_dict['rois'])

        if batch_dict['batch_type'] == 'merge':
            if dual_dict_flag:
                if 'rois_before_aug' in batch_dict_other:
                    rois_other = batch_dict_other['rois_before_aug'].clone().detach() # teacher model's rois
                else:
                    rois_other = batch_dict_other['rois'].clone().detach() # teacher model's rois
                rois_other = common_utils.reverse_augmentation(rois_other, batch_dict_other)
                rois = common_utils.forward_augmentation(rois_other, batch_target) # resize to student's scale
                batch_target['rois'] = rois
                batch_target['rois'] = self.augment_rois(batch_target['rois'])
        
        # merge source and target
        if batch_dict['batch_type'] == 'merge':
            batch_dict = self.merge_batch_dicts(batch_source, batch_target)

        pooled_features = self.roipool3d_gpu(batch_dict)  # (total_rois, num_sampled_points, 3 + C)

        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3)
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        shared_features = l_features[-1]

        # save the feature for feature alignment
        batch_dict['roi_head_features'] = shared_features

        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
                )
        batch_dict['batch_cls_preds'] = batch_cls_preds
        batch_dict['batch_box_preds'] = batch_box_preds
        batch_dict['cls_preds_normalized'] = False

        batch_dict['rcnn_cls'] = rcnn_cls
        batch_dict['rcnn_reg'] = rcnn_reg

        # pass to forward ret for source data
        if batch_dict['batch_type'] == 'merge':
            source_length = rcnn_cls.shape[0] // 2
            targets_dict['rcnn_cls'] = rcnn_cls[:source_length]
            targets_dict['rcnn_reg'] = rcnn_reg[:source_length]
            self.forward_ret_dict = targets_dict

        return batch_dict

            

        

