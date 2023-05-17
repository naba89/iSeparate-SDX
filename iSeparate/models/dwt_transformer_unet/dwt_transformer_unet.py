"""Copyright: Nabarun Goswami (2023)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWT1DForward, DWT1DInverse

from iSeparate.datasets.augmentation_modules import Augmenter
from iSeparate.models.base_model import BaseModel
from iSeparate.models.common.modules import Encoder, Decoder
from iSeparate.models.common.transformer import CrossTransformerEncoder
from iSeparate.models.common.utils import rescale_module


class DWTTransformerUNet(BaseModel):
    def __init__(self, audio_channels, target_sources, mixing_sources,
                 levels, wave, wavelet_enc_channels_dwt,
                 time_enc_channels, encoder_params,
                 decoder_params, cross_transformer_params, hidden_size_tsfmr,
                 augmentations, rescale, use_output_filter,
                 norm_starts, lstm_starts, attn_starts, wavelet_aug,
                 independent_post_filter=False):
        super().__init__()
        self.audio_channels = audio_channels
        self.mixing_sources = list(sorted(mixing_sources))
        self.target_sources = list(sorted(target_sources))
        self.target_indices = [self.mixing_sources.index(s) for s in self.target_sources]
        self.num_sources = len(self.target_sources)
        self.augmenter = Augmenter(augmentations)
        self.wavelet_aug = wavelet_aug

        self.levels = levels

        # 22-11, 11-5.5, 5.5-2.75, 2.75-1.375, 1.375-0.6875, 0.6875-0.34375, 0.34375-0.171875, 0.171875-0.0859375

        self.dwt = DWT1DForward(J=levels, wave=wave, mode='periodization')
        self.idwt = DWT1DInverse(wave=wave, mode='periodization')
        self.num_steps = levels + 1
        self.wavelet_encoders = nn.ModuleList()
        self.wavelet_decoders = nn.ModuleList()
        if len(wavelet_enc_channels_dwt) == 1:
            wavelet_enc_channels_dwt = wavelet_enc_channels_dwt * self.num_steps

        for i in range(self.num_steps):
            self.wavelet_encoders.append(Encoder(False, self.audio_channels, wavelet_enc_channels_dwt[i],
                                                 hidden_size_tsfmr, encoder_params, groups=1,
                                                 norm_starts=norm_starts,
                                                 lstm_starts=lstm_starts, attn_starts=attn_starts))
            self.wavelet_decoders.append(
                Decoder(False, wavelet_enc_channels_dwt[i], self.num_sources * self.audio_channels,
                        hidden_size_tsfmr, decoder_params, groups=1,
                        norm_starts=norm_starts,
                        lstm_starts=lstm_starts, attn_starts=attn_starts))

        self.t_encoder = Encoder(False, self.audio_channels, time_enc_channels,
                                 hidden_size_tsfmr, encoder_params, groups=1, norm_starts=norm_starts,
                                 lstm_starts=lstm_starts, attn_starts=attn_starts)
        self.t_decoder = Decoder(False, time_enc_channels, self.num_sources * self.audio_channels,
                                 hidden_size_tsfmr, decoder_params, groups=1, norm_starts=norm_starts,
                                 lstm_starts=lstm_starts, attn_starts=attn_starts)

        self.cross_transformer = CrossTransformerEncoder(dim=hidden_size_tsfmr,
                                                         **cross_transformer_params)

        self.use_output_filter = use_output_filter
        if self.use_output_filter:
            self.output_filter = nn.Conv1d((self.num_sources + 1) * self.audio_channels,
                                           self.num_sources * self.audio_channels,
                                           kernel_size=1, stride=1, padding=0, bias=False)

        self.independent_post_filter = independent_post_filter
        if self.independent_post_filter:
            self.post_filter = nn.ModuleList()
            for i in range(self.num_sources):
                self.post_filter.append(nn.ModuleList([
                    nn.Sequential(
                        nn.GELU(),
                        nn.Conv1d(self.audio_channels, 2 * self.audio_channels, kernel_size=15, stride=1, padding=7,
                                  bias=False)),
                    nn.Sequential(
                        nn.GELU(),
                        nn.Conv1d(2 * self.audio_channels, 2 * self.audio_channels, kernel_size=7, stride=1, padding=3,
                                  bias=False)),
                    nn.Sequential(
                        nn.GELU(),
                        nn.Conv1d(2 * self.audio_channels, self.audio_channels, kernel_size=3, stride=1, padding=1,
                                  bias=False))]
                ))

        if rescale:
            rescale_module(self, reference=rescale)

    def process_mix(self, x, return_postfiler_features=False):

        b, c, t = x.shape

        # collapse channels to batch dimension
        x = rearrange(x, 'b c t -> (b c) 1 t')

        pad_end = ((2 ** self.levels) * math.ceil(x.shape[-1] / (2 ** self.levels)) - x.shape[-1])
        x = F.pad(x, (0, pad_end), mode='constant', value=0)

        # normalize time inputs
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True)
        xt_normed = (x - mean) / (std + 1e-5)

        # time encoder
        time_inputs = xt_normed
        time_tsfmr_inp, skips_t, lengths_t = self.t_encoder(time_inputs)

        # wavelet transform
        with torch.cuda.amp.autocast(enabled=False):
            l_x, h_x = self.dwt(xt_normed.float())
        wavelet_inputs = [l_x] + h_x[::-1]

        if self.training and self.wavelet_aug:
            # drop out one band and replace with zeros,
            # randomly choose which band to drop and which batch items to apply it to
            # randomly choose 20% of batch indices
            batch_indices = torch.randperm(b)[:int(b * 0.2)]
            # randomly choose a band
            band_indices = torch.randperm(self.num_steps)[:1]
            # zero out the band for the batch indices
            for i in batch_indices:
                for j in band_indices:
                    wavelet_inputs[j][i] = wavelet_inputs[j][i] * 0

        # wavelet encoder
        wavelet_outputs = []
        wavelet_skips = []
        wavelet_lengths = []
        for i in range(self.num_steps):
            wavelet_outs, skips, lengths = self.wavelet_encoders[i](wavelet_inputs[i])
            wavelet_outputs.append(wavelet_outs)
            wavelet_skips.append(skips)
            wavelet_lengths.append(lengths)
        wavelet_tsfmr_inp = torch.stack(wavelet_outputs, dim=-1)

        # cross transformer
        wavelet_tsfmr_out, time_tsfmr_out = self.cross_transformer(wavelet_tsfmr_inp, time_tsfmr_inp)

        # wavelet decoder
        wavelet_outputs = list(torch.split(wavelet_tsfmr_out, 1, dim=-1))
        for i in range(self.num_steps):
            wavelet_outputs[i] = self.wavelet_decoders[i](wavelet_outputs[i].squeeze(-1), wavelet_skips[i],
                                                          wavelet_lengths[i])

        # inverse wavelet transform
        with torch.cuda.amp.autocast(enabled=False):
            out_dwt = [o.float() for o in wavelet_outputs]
            out_idwt = self.idwt([out_dwt[0], out_dwt[1:][::-1]])

        # time decoder
        out_time = self.t_decoder(time_tsfmr_out, skips_t, lengths_t)

        # combine outputs
        y_hat = out_idwt + out_time

        # apply post filter

        post_filter_fms = []
        if self.independent_post_filter:
            y_hat = rearrange(y_hat, 'b (s c) t -> b s c t', s=self.num_sources)
            y_hat_srcs = []
            for i in range(self.num_sources):
                fms = []
                out = y_hat[:, i]
                for j in range(len(self.post_filter[i])):
                    out = self.post_filter[i][j](out)
                    fms.append(out)
                y_hat_srcs.append(out)
                post_filter_fms.append(fms)
                # y_hat_srcs.append(self.post_filter[i](y_hat[:, i]))
            y_hat = torch.stack(y_hat_srcs, dim=1)
            y_hat = rearrange(y_hat, 'b s c t -> b (s c) t')

        # apply output filter
        if self.use_output_filter:
            y_hat = torch.cat([y_hat, xt_normed], dim=1)
            y_hat = self.output_filter(y_hat)

        y_hat = y_hat * std + mean

        # reshape
        y_hat = rearrange(y_hat, 'b (s c) t -> b s c t', s=self.num_sources)

        # unpad
        if pad_end > 0:
            y_hat = y_hat[..., :-pad_end]

        y_hat = rearrange(y_hat, '(b c) s 1 t -> b s c t', c=c)

        if return_postfiler_features:
            return y_hat, post_filter_fms

        return y_hat

    def forward(self, sources, augment=True):
        if self.training and augment:
            sources = self.augmenter(sources)
        x = sources.sum(dim=1)
        y = sources[:, self.target_indices]
        _y_hat, fms = self.process_mix(x, return_postfiler_features=True)
        output = {
            'y_hat': _y_hat,
            'post_filter_fms': fms,
        }

        targets = {
            'y': y,
        }

        return output, targets, sources
