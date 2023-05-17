"""Copyright: Nabarun Goswami (2023)."""
import itertools

import numpy as np
import torch
import torch.nn as nn
from asteroid.losses import MixITLossWrapper
from einops import rearrange
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from iSeparate.losses.losses import l1_loss


class SingleSrcL1(_Loss):
    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, *], got {targets.size()} and {est_targets.size()} instead"
            )
        # need to return [batch,]
        return torch.mean(torch.abs(est_targets - targets), dim=list(range(1, targets.ndim)))


class MixITnL1nMeanTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixit_loss_2mix = MixITLossWrapper(SingleSrcL1(), generalized=True)
        self.l1 = l1_loss

    def forward(self, outputs, targets, ema_model=None, weights=None, mean_teacher_enable=False):

        if weights is None:
            weights = [1, 0, 0]

        loss = 0
        if weights[0] != 0:
            loss += weights[0] * self.l1(outputs, targets)

        if weights[1] != 0:
            output = outputs['y_hat']
            y = targets['y']

            perm = torch.randperm(y.shape[1])
            y_mix_1 = perm[:2]
            y_mix_2 = perm[2:]
            y_tar_1 = y[:, y_mix_1, ...].sum(dim=1)
            y_tar_2 = y[:, y_mix_2, ...].sum(dim=1)
            target = torch.stack([y_tar_1, y_tar_2], dim=1)

            loss += weights[1] * self.mixit_loss_2mix(output, target)

        if weights[2] != 0 and ema_model is not None and mean_teacher_enable:
            with torch.inference_mode():
                y = targets['y']
                num_sources = y.shape[1]
                y = rearrange(y.detach(), 'b s c t -> (b s) c t')
                y_reg_tar = ema_model.separate(y, patch_length=None)  # b k c t
                y_reg = rearrange(y_reg_tar, '(b s) k c t -> b s k c t', s=num_sources)
                y_reg_tar = y_reg.sum(dim=1)
                targets_reg = {'y': y_reg_tar}
            loss += weights[2] * self.l1(outputs, targets_reg)

        loss = loss / sum(weights)
        return loss


class MixITnL1nMeanTeacherV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixit_loss_2mix = MixITLossWrapper(SingleSrcL1(), generalized=True)
        self.l1 = l1_loss
        self.vanilla_l1_loss = nn.L1Loss()

    def forward(self, outputs, targets, ema_model=None, weights=None, mean_teacher_enable=False):

        if weights is None:
            weights = [1, 1, 0, 0, 0]

        divisor = 0
        loss = 0
        if weights[0] != 0:
            loss += weights[0] * self.l1(outputs, targets)
            divisor += weights[0]

        if weights[1] != 0:
            output = outputs['y_hat'].sum(dim=1)
            target = targets['y'].sum(dim=1)
            loss += weights[1] * self.vanilla_l1_loss(output, target)
            divisor += weights[1]

        if weights[2] != 0:
            output = outputs['y_hat']
            y = targets['y']
            perm = torch.randperm(y.shape[1])
            y_mix_1 = perm[:2]
            y_mix_2 = perm[2:]
            y_tar_1 = y[:, y_mix_1, ...].sum(dim=1)
            y_tar_2 = y[:, y_mix_2, ...].sum(dim=1)
            target = torch.stack([y_tar_1, y_tar_2], dim=1)

            loss += weights[2] * self.mixit_loss_2mix(output, target)
            divisor += weights[2]

        if weights[3] != 0 and ema_model is not None and mean_teacher_enable:
            with torch.inference_mode():
                output = outputs['y_hat']
                ema_inp = output.detach().sum(dim=1)
                ema_out = ema_model.separate(ema_inp, patch_length=None)
                targets_reg = {'y': ema_out}

            loss += weights[3] * self.l1(outputs, targets_reg)
            divisor += weights[3]

        loss = loss / divisor
        return loss
