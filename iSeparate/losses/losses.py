"""Copyright: Nabarun Goswami (2023)."""
import torch
import torch.nn as nn


def bsrnn_loss(outputs, targets, weights=None):
    y = targets['y']
    y_hat = outputs['y_hat']

    spec_y = targets['y_freq']
    spec_y_hat = outputs['y_hat_freq']

    if weights is None:
        weights = y_hat.new_ones(y_hat.shape[1])

    loss = l1_loss(y_hat, y, weights=weights)
    loss += l1_loss(spec_y_hat.real, spec_y.real, weights=weights)
    loss += l1_loss(spec_y_hat.imag, spec_y.imag, weights=weights)

    return loss


def l1_loss(output, target, weights=None):
    if isinstance(output, dict):
        output = output['y_hat'].float()
    if isinstance(target, dict):
        target = target['y'].float()

    if weights is None:
        weights = output.new_ones(output.shape[1])

    reduction_dims = tuple(range(2, output.dim()))
    loss = nn.L1Loss(reduction="none")(output, target)

    loss = loss.mean(dim=reduction_dims).mean(0)
    loss = (loss * weights).sum() / weights.sum()

    return loss


def global_sdr(outputs, targets):
    if isinstance(outputs, dict):
        separations = outputs['y_hat']  # (batch, nsrc, nchan, nframes)
        references = targets['y']
    else:
        separations = outputs
        references = targets
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(references ** 2, dim=(2, 3))
    den = torch.sum((references - separations) ** 2, dim=(2, 3))
    num += delta
    den += delta
    sdr = -10 * torch.log10(num / den)
    return sdr.mean(0)  # (nsrc,)

