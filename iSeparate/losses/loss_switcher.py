import torch


def get_loss_fn(loss_name):

    if loss_name == "l1_loss":
        from iSeparate.losses.losses import l1_loss
        return l1_loss
    elif loss_name == "l1_with_mean_teacher":
        from iSeparate.losses.pairwise import MixITnL1nMeanTeacher
        return MixITnL1nMeanTeacher()
    elif loss_name == "l1_with_mean_teacher_v2":
        from iSeparate.losses.pairwise import MixITnL1nMeanTeacherV2
        return MixITnL1nMeanTeacherV2()
    elif loss_name == "bsrnn_l1":
        from iSeparate.losses.losses import bsrnn_loss
        return bsrnn_loss


def get_eval_metric(loss_name):
    if loss_name == 'global_sdr':
        from iSeparate.losses.losses import global_sdr
        metric = global_sdr
    else:
        metric = torch.nn.L1Loss()

    return metric
