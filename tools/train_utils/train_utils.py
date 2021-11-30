import glob
import os

import torch
import tqdm
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pdb

def update_ema_variables(model, ema_model, model_cfg=None, cur_epoch=None, total_epochs=None, cur_it=None, total_it=None):
    assert model_cfg is not None

    multiplier = 1.0

    alpha = model_cfg['EMA_MODEL_ALPHA']
    alpha = 1 - multiplier*(1-alpha)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    if model_cfg.get('COPY_BN_STATS_TO_TEACHER', False):
        if model_cfg.get('BN_WARM_UP', False) and cur_epoch==0:
            multiplier = (np.cos(cur_it/total_it*np.pi) + 1) * 0.5
            bn_ema = model_cfg.BN_EMA - multiplier * (model_cfg.BN_EMA - 0.9)
            if hasattr(model, 'module'):
                model.module.set_momemtum_value_for_bn(momemtum=(1-bn_ema))
            else:
                model.set_momemtum_value_for_bn(momemtum=(1-bn_ema))

        model_named_buffers = model.module.named_buffers() if hasattr(model, 'module') else model.named_buffers()
        
        for emabf, bf in zip(ema_model.named_buffers(), model_named_buffers):
            emaname, emavalue = emabf
            name, value = bf
            assert emaname == name, 'name not equal:{} , {}'.format(emaname, name)
            if 'running_mean' in name or 'running_var' in name:
                emavalue.data = value.data


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    train_loader_target=None, train_loader_source=None, ema_model=None, cur_epoch=None, total_epochs=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    
    '''target loader'''
    if train_loader_target:
        dataloader_iter_target = iter(train_loader_target)
    if train_loader_source:
        dataloader_iter_source = iter(train_loader_source)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        ''' target loader'''
        if train_loader_target:
            try:
                batch_target = next(dataloader_iter_target)
            except StopIteration:
                dataloader_iter_target = iter(train_loader_target)
                batch_target = next(dataloader_iter_target)
            batch = [batch, batch_target]

        '''source loader for mt'''
        if train_loader_source:
            try:
                batch_source = next(dataloader_iter_source)
            except StopIteration:
                dataloader_iter_source = iter(train_loader_source)
                batch_source = next(dataloader_iter_source)
            assert isinstance(batch, list)
            batch.append(batch_source)

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        if ema_model != None:
            loss, tb_dict, disp_dict = model_func(model, batch, ema_model=ema_model, cur_epoch=cur_epoch)
        else:
            loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        if ema_model != None:
            update_ema_variables(model, ema_model, model_cfg=ema_model.model_cfg, cur_epoch=cur_epoch, total_epochs=total_epochs,
                                cur_it=cur_it, total_it=total_it_each_epoch)

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False,
                train_loader_target=None, train_sampler_target=None,
                train_loader_source=None, train_sampler_source=None,
                ema_model=None):

    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            # for target dataset
            if train_sampler_target is not None:
                train_sampler_target.set_epoch(cur_epoch)
            # for source dataset (mt)
            if train_sampler_source is not None:
                train_sampler_source.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            if ema_model and ema_model.model_cfg.get('BN_EMA_DECAY', False):
                max_bn_ema = ema_model.model_cfg.BN_EMA
                min_bn_ema = ema_model.model_cfg.MIN_BN_EMA
                multiplier = (np.cos(cur_epoch/total_epochs*np.pi) + 1) * 0.5
                cur_bn_ema = min_bn_ema + multiplier * (max_bn_ema - min_bn_ema)
                model.module.set_momemtum_value_for_bn(momemtum=(1-cur_bn_ema))
                # print('bn momemtum set as {}'.format(cur_bn_ema))

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                train_loader_target=train_loader_target,
                train_loader_source=train_loader_source,
                ema_model=ema_model,
                cur_epoch=cur_epoch,
                total_epochs=total_epochs
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
