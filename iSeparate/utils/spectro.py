
"""Conveniance wrapper to perform STFT and iSTFT
This code is adapted from https://github.com/facebookresearch/demucs
with original license at https://github.com/facebookresearch/demucs/blob/main/LICENSE
"""

import torch as th

from iSeparate.utils.stft import istft


def spectro(x, n_fft=512, hop_length=None, pad=0, window=None):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=window if window is not None else th.hann_window(n_fft, dtype=x.dtype, device=x.device),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0, window=None, skip_nola=True):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    if window is not None and window.shape[0] != win_length:
        window = th.hann_window(win_length,
                                dtype=z.real.dtype, device=z.device)
    x = istft(z,
              n_fft,
              hop_length,
              window=window if window is not None else th.hann_window(win_length,
                                                                      dtype=z.real.dtype, device=z.device),
              win_length=win_length,
              normalized=True,
              length=length,
              center=True,
              skip_nola=skip_nola)
    _, length = x.shape
    return x.view(*other, length)
