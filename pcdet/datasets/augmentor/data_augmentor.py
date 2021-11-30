from functools import partial
import torch
import random
import numpy as np
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, box_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def object_size_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.object_size_normalization, config=config)
        
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        if gt_boxes.shape[1] > 7:
            gt_boxes = gt_boxes[:,:7]
        offset = np.array(config['OFFSET'])
        # get masks of points inside boxes
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)).numpy()

        num_obj = gt_boxes.shape[0]
        obj_points_list = []
        gt_boxes_size = gt_boxes[:, 3:6]
        new_gt_boxes_size = gt_boxes_size + offset
        scale_factor = new_gt_boxes_size / gt_boxes_size
        # scale the objects
        for i in range(num_obj):
            point_mask = point_masks[i]
            obj_points = points[point_mask > 0] # get object points within the gt box
            obj_points[:, :3] -= gt_boxes[i, :3] # relative to box center
            obj_points[:, :3] *= scale_factor[i] # scale
            obj_points[:, :3] += gt_boxes[i, :3] # back to global coordinate
            obj_points_list.append(obj_points)

        # remove points inside boxes
        points = box_utils.remove_points_in_boxes3d(points, gt_boxes)
        # scale the boxes
        gt_boxes[:, 3:6] *= scale_factor
        # remove points inside boxes
        points = box_utils.remove_points_in_boxes3d(points, gt_boxes)

        # merge points
        # points = box_utils.remove_points_in_boxes3d(points, gt_boxes)
        obj_points = np.concatenate(obj_points_list, axis=0)
        points = np.concatenate([points, obj_points], axis=0)

        data_dict['points'] = points
        data_dict['gt_boxes'][:,:7] = gt_boxes
        return data_dict
    
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None
        points = data_dict['points']

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            if 'gt_boxes' in data_dict:
                gt_boxes, points, world_flip_enabled = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points, return_enable=True
                )
            else:
                points, world_flip_enabled = getattr(augmentor_utils, 'random_flip_along_%s_points' % cur_axis)(
                    points, return_enable=True
                )
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['world_flip_enabled'] = world_flip_enabled
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        if 'gt_boxes' in data_dict:
            gt_boxes, points, world_rotation = augmentor_utils.global_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rotation=True
            )
        else:
            points, world_rotation = augmentor_utils.global_rotation_points(
                data_dict['points'], rot_range=rot_range, return_rotation=True
            )

        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['world_rotation'] = world_rotation
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        if 'gt_boxes' in data_dict:
            gt_boxes, points, scale_ratio = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
            )
        else:
            points, scale_ratio = augmentor_utils.global_scaling_points(data_dict['points'], config['WORLD_SCALE_RANGE'])
            
        data_dict['world_scaling'] = scale_ratio
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling_xyz(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling_xyz, config=config)
        gt_boxes = data_dict['gt_boxes']
        points = data_dict['points']
        scale_range = config['SCALE_RANGE']
        noise_scale = np.random.uniform(scale_range[0], scale_range[1], 3)
        points[:, :3] *= noise_scale
        gt_boxes[:, :3] *= noise_scale
        gt_boxes[:, 3:6] *= noise_scale
        data_dict['points'] = points
        data_dict['gt_boxes'] = gt_boxes
        data_dict['world_scaling_xyz'] = noise_scale
        return data_dict

    def jitter_point_cloud(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.jitter_point_cloud, config=config)
        sigma = config['SIGMA']
        clip = config['CLIP']
        assert(clip > 0)
        points = data_dict['points']
        jittered_data = np.clip(sigma * np.random.randn(points.shape[0], points.shape[1]), -1*clip, clip)
        points += jittered_data
        data_dict['points'] = points
        data_dict['jittered'] = True
        data_dict['jitter_values'] = jittered_data
        return data_dict

    def random_world_shift(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_shift, config=config)
        shift_range = config['RANGE']
        shifts = np.random.uniform(-shift_range, shift_range, 3)
        data_dict['points'] += shifts
        data_dict['world_shifts'] = shifts
        return data_dict

    def forward(self, data_dict, augment=True):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if augment:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes' in data_dict and 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
