from collections import namedtuple

import numpy as np
import torch
import numpy as np
from .detectors import build_detector


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    if isinstance(batch_dict, dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
                continue
            batch_dict[key] = torch.from_numpy(val).float().cuda()
    else:
        assert isinstance(batch_dict, list)
        for batch in batch_dict:
            for key, val in batch.items():
                if not isinstance(val, np.ndarray):
                    continue
                if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
                    continue
                batch[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


def model_fn_decorator_for_mt():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, ema_model=None, cur_epoch=None):

        batch_source, batch_target = batch_dict
        load_data_to_gpu(batch_source)
        load_data_to_gpu(batch_target)
        batch_target1, batch_target2 = batch_target

        # add tag for target domain
        batch_target1['is_target_domain'] = True
        batch_target2['is_target_domain'] = True
        
        # forward teacher model first
        batch_target2 = ema_model(batch_target2, is_ema=True)

        # forward main model
        ret_dict, tb_dict, disp_dict = model([batch_source, batch_target1, batch_target2], is_ema=False, cur_epoch=cur_epoch)

        # pdb.set_trace()

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

def model_fn_decorator_for_mt_merge_source_target():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, ema_model=None, cur_epoch=None):

        batch_source, batch_target = batch_dict
        load_data_to_gpu(batch_source)
        load_data_to_gpu(batch_target)
        batch_target1, batch_target2 = batch_target

        batch_merge = merge_batch_dicts(batch_source, batch_target1)

        # add tag for target domain
        batch_merge['batch_type'] = 'merge' # merged source and target
        batch_target2['batch_type'] = 'target'
        # batch_target2['is_target_domain'] = True
        
        # forward teacher model first
        batch_target2 = ema_model(batch_target2, is_ema=True)
        # forward student model
        ret_dict, tb_dict, disp_dict = model([batch_merge, batch_target2], is_ema=False, cur_epoch=cur_epoch, ema_model=ema_model)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

def model_fn_decorator_for_mt_merge_both_models():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, ema_model=None, cur_epoch=None):

        batch_source, batch_target = batch_dict
        load_data_to_gpu(batch_source)
        load_data_to_gpu(batch_target)
        batch_source1, batch_source2 = batch_source
        batch_target1, batch_target2 = batch_target

        batch_student = merge_batch_dicts(batch_source1, batch_target1)
        batch_teacher = merge_batch_dicts(batch_source2, batch_target2)

        batch_student['batch_type'] = 'student'
        batch_teacher['batch_type'] = 'teacher'

        # forward teacher model first
        batch_teacher = ema_model(batch_teacher, is_ema=True)

        # forward main model
        ret_dict, tb_dict, disp_dict = model([batch_student, batch_teacher], is_ema=False, cur_epoch=cur_epoch)

        # pdb.set_trace()

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

def merge_batch_dicts(source, target):
    output = {}
    for key, val1 in source.items():
        if key == 'batch_size':
            output[key] = val1 * 2
        elif key == 'gt_boxes':
            output[key] = val1
        elif key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            if key in target:
                output[key] = np.concatenate([val1, target[key]], axis=0)
        else:
            if key in target:
                val2 = target[key]
                if key == 'points':
                    val2[:,0] += target['batch_size']
                output[key] = torch.cat([val1, val2], dim=0)
            else:
                if key == 'world_flip_enabled' or key == 'world_rotation':
                    val2 = torch.zeros_like(val1)
                elif key == 'world_scaling':
                    val2 = torch.ones_like(val1)
                output[key] = torch.cat([val1, val2], dim=0)
    return output

