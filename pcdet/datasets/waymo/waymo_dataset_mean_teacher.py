# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import copy
import numpy as np
from ...utils import box_utils, common_utils
from ..waymo.waymo_dataset import WaymoDataset
from collections import defaultdict


class WaymoDatasetMeanTeacher(WaymoDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.include_waymo_data(self.mode)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if self.dataset_cfg.PROVIDE_GT:
            if 'annos' in info:
                annos = info['annos']
                annos = common_utils.drop_info_with_name(annos, name='unknown')

                if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                    gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
                else:
                    gt_boxes_lidar = annos['gt_boxes_lidar']

                input_dict.update({
                    'gt_names': annos['name'],
                    'gt_boxes': gt_boxes_lidar,
                    'num_points_in_gt': annos.get('num_points_in_gt', None)
                })

        if self.dataset_cfg.SAME_POINT_SAMPLING:
            # perform gt sampling here before point sampling to ensure fixed number of points
            if self.pre_data_augmentor:
                gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
                input_dict['gt_boxes_mask'] = gt_boxes_mask
                input_dict = self.pre_data_augmentor.forward(data_dict=input_dict)

            data_dict1 = self.process_data(input_dict)
            data_dict2 = copy.deepcopy(data_dict1)
            augment_student = self.dataset_cfg.AUGMENT_STUDENT
            augment_teacher = self.dataset_cfg.AUGMENT_TEACHER
            data_dict1 = self.prepare_data(data_dict=data_dict1, augment=augment_student, process_data=False)
            data_dict2 = self.prepare_data(data_dict=data_dict2, augment=augment_teacher, process_data=False)
            data_dict1['metadata'] = info.get('metadata', info['frame_id'])
            data_dict2['metadata'] = info.get('metadata', info['frame_id'])
            data_dict1.pop('num_points_in_gt', None)
            data_dict2.pop('num_points_in_gt', None)
        else:
            raise NotImplementedError
        
        output = [data_dict1, data_dict2]
        return output

    def prepare_data(self, data_dict, augment=True, process_data=True):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in) # lidar points
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...] # lidar boxes
                gt_names: optional, (N), string
                process_data: if False, pass in processed data
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            if self.dataset_cfg.PROVIDE_GT:
                assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
                gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            else:
                gt_boxes_mask = None

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                },
                augment=augment
            )

            if 'gt_boxes' not in data_dict:
                data_dict.pop('gt_boxes_mask', None)

        if self.dataset_cfg.PROVIDE_GT and data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        
        if process_data:
            data_dict = self.point_feature_encoder.forward(data_dict)
            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )

        data_dict.pop('gt_names', None)

        return data_dict

    def process_data(self, data_dict):
        data_dict = self.point_feature_encoder.forward(data_dict)
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        return data_dict


    @staticmethod
    def collate_batch(batch_list, _unused=False):
        def collate_fn(batch_list):
            data_dict = defaultdict(list)
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)
            ret = {}

            for key, val in data_dict.items():
                try:
                    if key in ['voxels', 'voxel_num_points']:
                        ret[key] = np.concatenate(val, axis=0)
                    elif key in ['points', 'voxel_coords']:
                        coors = []
                        for i, coor in enumerate(val):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(coor_pad)
                        ret[key] = np.concatenate(coors, axis=0)
                    elif key in ['gt_boxes']:
                        max_gt = max([len(x) for x in val])
                        batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                        for k in range(batch_size):
                            batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                        ret[key] = batch_gt_boxes3d
                    else:
                        ret[key] = np.stack(val, axis=0)
                except:
                    print('Error in collate_batch: key=%s' % key)
                    raise TypeError

            ret['batch_size'] = batch_size
            return ret

        if isinstance(batch_list[0], dict):
            return collate_fn(batch_list)

        else:
            assert isinstance(batch_list[0], list)
            batch_list1 = [x[0] for x in batch_list]
            batch_list2 = [x[1] for x in batch_list]
            ret1 = collate_fn(batch_list1)
            ret2 = collate_fn(batch_list2)
            return [ret1, ret2]

   