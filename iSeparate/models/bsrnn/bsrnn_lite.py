"""!

@author Yi Luo (oulyluo)
@copyright Tencent AI Lab

This code is adapted from the BSRNN baseline
https://gitlab.aicrowd.com/Tomasyu/sdx-2023-music-demixing-track-starter-kit
"""

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from iSeparate.datasets.augmentation_modules import Augmenter
from iSeparate.models.base_model import BaseModel


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0],
                                                                                           input.shape[2],
                                                                                           input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = ResRNN(self.feature_dim, self.feature_dim * 2)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim * 2)

    def forward(self, input):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(input.view(B * self.nband, self.feature_dim, -1)).view(B, self.nband, -1, T)

        # band comm
        band_output = band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.nband)
        output = self.band_comm(band_output).view(B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous()

        return output.view(B, N, T)


class Separator(nn.Module):
    def __init__(self, sr=44100, win=2048, stride=512, feature_dim=80, num_repeat=10, instrument='vocals'):
        super().__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.register_buffer('window', torch.hann_window(self.win))
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_50 = int(np.floor(50 / (sr / 2.) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sr / 2.) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sr / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sr / 2.) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.) * self.enc_dim))

        # if instrument == 'vocals' or instrument == 'other' or instrument == 'speech':
        # 0-500: 50, 500-1k: 100, 1k-4k: 250, 4k-8k: 500, 8k-16k: 1k, 16k-inf
        if instrument in ['vocals', 'other', 'bass', "drums"]:
            # self.band_width = [bandwidth_50] * 10
            # self.band_width = [bandwidth_100] * 5
            # self.band_width += [bandwidth_250] * 12
            # self.band_width += [bandwidth_500] * 8
            # self.band_width += [bandwidth_1k] * 8
            self.band_width = [bandwidth_100] * 10
            self.band_width += [bandwidth_250] * 12
            self.band_width += [bandwidth_500] * 8
        elif instrument in ['speech', 'music', 'sfx']:
            self.band_width = [bandwidth_100] * 10
            self.band_width += [bandwidth_250] * 12
            self.band_width += [bandwidth_500] * 8
        else:
            print("Unknown Instrument {}".format(instrument))
            raise NotImplementedError

        self.band_width.append(self.enc_dim - np.sum(self.band_width))

        self.nband = len(self.band_width)
        # print(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                                         nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1)
                                         )
                           )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(BSNet(self.nband * self.feature_dim, self.nband))
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(nn.Sequential(nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                                           nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                                           nn.Tanh(),
                                           nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1)
                                           )
                             )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input, return_spec=False):
        # input shape: (B, C, T)

        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)

        # frequency-domain separation
        spec = torch.stft(input, n_fft=self.win, hop_length=self.stride,
                          window=self.window,
                          return_complex=True)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx:band_idx + self.band_width[i]].contiguous())
            subband_mix_spec.append(spec[:, band_idx:band_idx + self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(self.BN[i](subband_spec[i].view(batch_size * nch, self.band_width[i] * 2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        # import pdb; pdb.set_trace()
        # separator
        sep_output = self.separator(
            subband_feature.view(batch_size * nch, self.nband * self.feature_dim, -1))  # B, nband*N, T
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(batch_size * nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = subband_mix_spec[i].real * this_mask_real - subband_mix_spec[
                i].imag * this_mask_imag  # B*nch, BW, T
            est_spec_imag = subband_mix_spec[i].real * this_mask_imag + subband_mix_spec[
                i].imag * this_mask_real  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T

        output = torch.istft(est_spec.view(batch_size * nch, self.enc_dim, -1),
                             n_fft=self.win, hop_length=self.stride,
                             window=self.window, length=nsample)

        output = output.view(batch_size, nch, -1)
        if return_spec:
            est_spec = est_spec.view(batch_size, nch, self.enc_dim, -1)
            return output, est_spec
        return output


class BSRNN(BaseModel):
    def __init__(self, mixing_sources, target_sources, sr=44100,
                 win=2048, stride=512, feature_dim=80, num_repeat=10,
                 augmentations=None):
        super(BSRNN, self).__init__()

        self.augmenter = Augmenter(augmentations)

        self.sr = sr
        self.win = win
        self.stride = stride
        self.feature_dim = feature_dim
        self.num_repeat = num_repeat

        self.mixing_sources = list(sorted(mixing_sources))
        self.target_sources = list(sorted(target_sources))
        self.target_indices = [self.mixing_sources.index(s) for s in self.target_sources]
        self.num_sources = len(self.target_sources)

        self.separators = nn.ModuleList()
        for i, source in enumerate(self.target_sources):
            self.separators.append(Separator(sr=sr, win=win, stride=stride, feature_dim=feature_dim,
                                             num_repeat=num_repeat, instrument=source))

    def forward(self, sources):
        if self.training:
            sources = self.augmenter(sources)
        x = sources.sum(dim=1)
        y = sources[:, self.target_indices]
        b, s, c, t = y.shape
        y = rearrange(y, 'b s c t -> (b s c) t')
        y_freq = torch.stft(y, n_fft=self.win, hop_length=self.stride,
                            window=self.separators[0].window,
                            return_complex=True)
        y = rearrange(y, '(b s c) t -> b s c t', s=self.num_sources, b=b)
        y_freq = rearrange(y_freq, '(b s c) f t -> b s c f t', s=self.num_sources, b=b)

        time_outputs = []
        freq_outputs = []
        for i, source in enumerate(self.target_sources):
            output, freq_output = self.separators[i](x, return_spec=True)
            time_outputs.append(output)
            freq_outputs.append(freq_output)

        y_hat = torch.stack(time_outputs, dim=1)
        y_hat_freq = torch.stack(freq_outputs, dim=1)
        output = {
            'y_hat': y_hat,
            'y_hat_freq': y_hat_freq,
        }

        target = {
            'y': y,
            'y_freq': y_freq,
        }

        return output, target, sources

    def process_mix(self, mix):
        x = mix
        time_outputs = []
        for i, source in enumerate(self.target_sources):
            output = self.separators[i](x)
            time_outputs.append(output)

        y_hat = torch.stack(time_outputs, dim=1)

        return y_hat
