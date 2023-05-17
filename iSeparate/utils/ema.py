import copy

import torch


class ModelEMA2:
    """
    Perform EMA on a model.
    """
    def __init__(self, model, decay=0.9999, unbias=True, device='cpu'):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.device = device
        self.decay = decay
        self.count = 0
        self.unbias = unbias
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval().to(device)

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def lazy_init(self, model):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval().to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        if self.unbias:
            self.count = self.count * self.decay + 1
            w = 1 / self.count
        else:
            w = 1 - self.decay

        for online_param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            try:
                ema_param.mul_(1 - w).add_(online_param.detach(), alpha=w)
            except RuntimeError:
                ema_param.copy_(online_param.detach(), non_blocking=True)

        for online_buffer, ema_buffer in zip(model.buffers(), self.ema_model.buffers()):
            ema_buffer.copy_(online_buffer.detach(), non_blocking=True)

    def state_dict(self):
        return {'state': self.ema_model.state_dict(), 'count': self.count}

    def load_state_dict(self, state):
        self.count = state.pop('count')
        self.ema_model.load_state_dict(state['state'])
