from torch.nn.modules.batchnorm import _BatchNorm # , BatchNorm2d as _BatchNorm2d
from torch.nn import functional as F
import torch
import os
import pdb

class BatchNorm2d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        # set an attribute only for teacher model
        self.momentum_update_for_teacher = False
        self.update_running = True

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        if self.momentum_update_for_teacher:
            assert self.update_running

            # pdb.set_trace()

            mean = torch.mean(x, dim=[0, 2, 3])
            var = torch.var(x, dim=[0, 2, 3])
            n = x.numel() / x.size(1)

            with torch.no_grad():
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                # update running_var with unbiased var
                self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var

            x = (x - self.running_mean[None, :, None, None].detach()) / (
                torch.sqrt(self.running_var[None, :, None, None].detach() + self.eps)
            )
            if self.affine:
                x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            return x
        
        elif not self.update_running:
            assert not self.momentum_update_for_teacher

            mean = torch.mean(x, dim=[0, 2, 3])
            var = torch.var(x, dim=[0, 2, 3])
            n = x.numel() / x.size(1)

            x = (x - mean[None, :, None, None].detach()) / (
                torch.sqrt(var[None, :, None, None].detach() + self.eps)
            )

            if self.affine:
                x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            return x

        else:
            return super().forward(x)


class BatchNorm1d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        # set an attribute only for teacher model
        self.momentum_update_for_teacher = False
        self.update_running = True
        self.save_mean_to_file = False

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        # do running mean update explicitly for teacher model
        if self.momentum_update_for_teacher:
            assert self.update_running
            num_dim = len(x.shape)
            if num_dim == 2:
                mean = torch.mean(x, dim=[0])
                var = torch.var(x, dim=[0])
            else:
                mean = torch.mean(x, dim=[0, 2])
                var = torch.var(x, dim=[0, 2])
            n = x.numel() / x.size(1)

            with torch.no_grad():
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                # update running_var with unbiased var
                self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var

            if num_dim == 2:
                x = (x - self.running_mean[None, :].detach()) / (torch.sqrt(self.running_var[None, :].detach() + self.eps))
            else:
                x = (x - self.running_mean[None, :, None].detach()) / (torch.sqrt(self.running_var[None, :, None].detach() + self.eps))

            
            if self.affine:
                if num_dim == 2:
                    x = x * self.weight[None, :] + self.bias[None, :]
                else:
                    x = x * self.weight[None, :, None] + self.bias[None, :, None]

            return x

        if not self.update_running:
            assert not self.momentum_update_for_teacher
            num_dim = len(x.shape)
            if num_dim == 2:
                mean = torch.mean(x, dim=[0])
                var = torch.var(x, dim=[0])
            else:
                mean = torch.mean(x, dim=[0, 2])
                var = torch.var(x, dim=[0, 2])
            n = x.numel() / x.size(1)

            if num_dim == 2:
                x = (x - mean[None, :].detach()) / (torch.sqrt(var[None, :].detach() + self.eps))
            else:
                x = (x - mean[None, :, None].detach()) / (torch.sqrt(var[None, :, None].detach() + self.eps))
            
            if self.affine:
                if num_dim == 2:
                    x = x * self.weight[None, :] + self.bias[None, :]
                else:
                    x = x * self.weight[None, :, None] + self.bias[None, :, None]

            return x

        else:
            # normal BN if not special case
            return super().forward(x)
