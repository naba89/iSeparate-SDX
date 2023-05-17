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
from iSeparate.models.common.utils import pad1d, rescale_module
from iSeparate.utils.spectro import spectro, ispectro


class WaveletHTDemucs(BaseModel):
    def __init__(self, audio_channels, target_sources, mixing_sources,
                 levels, wave, nfft, hop, freq_emb_weight,
                 hidden_sizes_wavelet, wavelet_enc_channels,
                 time_enc_channels, freq_enc_channels, encoder_params,
                 decoder_params, cross_transformer_params, hidden_size_f,
                 hidden_size_t, augmentations, rescale,
                 norm_starts, lstm_starts, attn_starts):
        super().__init__()
        self.audio_channels = audio_channels
        self.mixing_sources = list(sorted(mixing_sources))
        self.target_sources = list(sorted(target_sources))
        self.target_indices = [self.mixing_sources.index(s) for s in self.target_sources]
        self.num_sources = len(self.target_sources)
        self.augmenter = Augmenter(augmentations)

        self.levels = levels
        self.max_period = 10000
        self.sin_random_shift = 0
        self.dwt = DWT1DForward(J=levels, wave=wave, mode='periodization')
        self.idwt = DWT1DInverse(wave=wave, mode='periodization')

        self.hop = hop
        self.nfft = nfft
        self.register_buffer('window', torch.hann_window(nfft).float())
        # self.stft = STFT(filter_length=self.nfft, hop_length=self.hop, win_length=self.nfft)
        self.num_steps = levels + 1
        self.num_freqs = nfft // 2  # we ignore the last frequency bin
        self.freq_splits = list(reversed([self.num_freqs // 2 ** (i + 1) for i in range(levels)] + [
            self.num_freqs // 2 ** self.levels]))

        self.hidden_sizes_wavelet = hidden_sizes_wavelet
        self.wavelet_encoders = nn.ModuleList()
        self.wavelet_decoders = nn.ModuleList()
        if len(wavelet_enc_channels) == 1:
            wavelet_enc_channels = wavelet_enc_channels * self.num_steps
        if len(freq_enc_channels) == 1:
            freq_enc_channels = freq_enc_channels * self.num_steps
        if len(self.hidden_sizes_wavelet) == 1:
            self.hidden_sizes_wavelet = self.hidden_sizes_wavelet * self.num_steps
        for i in range(self.num_steps):
            self.wavelet_encoders.append(Encoder(False, self.audio_channels, wavelet_enc_channels[i],
                                                 self.hidden_sizes_wavelet[i], encoder_params, groups=1,
                                                 norm_starts=norm_starts,
                                                 lstm_starts=lstm_starts, attn_starts=attn_starts))
            self.wavelet_decoders.append(Decoder(False, wavelet_enc_channels[i], self.num_sources * self.audio_channels,
                                                 self.hidden_sizes_wavelet[i], decoder_params, groups=1,
                                                 norm_starts=norm_starts,
                                                 lstm_starts=lstm_starts, attn_starts=attn_starts))

        self.f_encoder = Encoder(True, self.audio_channels * 2, freq_enc_channels,
                                 hidden_size_f, encoder_params, groups=1,
                                 freq_emb_weight=freq_emb_weight,
                                 num_freqs=self.num_freqs, norm_starts=norm_starts,
                                 lstm_starts=lstm_starts, attn_starts=attn_starts)

        self.f_decoder = Decoder(True, freq_enc_channels, self.num_sources * self.audio_channels * 2,
                                 hidden_size_f, decoder_params, groups=1, norm_starts=norm_starts,
                                 lstm_starts=lstm_starts, attn_starts=attn_starts)

        self.t_encoder = Encoder(False, self.audio_channels, time_enc_channels,
                                 hidden_size_t, encoder_params, groups=1, norm_starts=norm_starts,
                                 lstm_starts=lstm_starts, attn_starts=attn_starts)
        self.t_decoder = Decoder(False, time_enc_channels, self.num_sources * self.audio_channels,
                                 hidden_size_t, decoder_params, groups=1, norm_starts=norm_starts,
                                 lstm_starts=lstm_starts, attn_starts=attn_starts)

        self.freq_branch_compress = nn.ModuleList([nn.Conv2d(freq_enc_channels[-1][0], self.hidden_sizes_wavelet[i],
                                                             kernel_size=1, stride=1, padding=0) for i in
                                                   range(self.num_steps)])
        self.freq_branch_decompress = nn.ModuleList([nn.Conv2d(self.hidden_sizes_wavelet[i], freq_enc_channels[-1][0],
                                                               kernel_size=1, stride=1, padding=0) for i in
                                                     range(self.num_steps)])

        self.cross_transformer = nn.ModuleList([CrossTransformerEncoder(dim=self.hidden_sizes_wavelet[i],
                                                                        **cross_transformer_params) for i in
                                                range(self.num_steps)])
        assert hidden_size_t == hidden_size_f
        self.cross_transformer_full = CrossTransformerEncoder(dim=hidden_size_t,
                                                              **cross_transformer_params)

        self.output_filter = nn.Conv1d((self.num_sources + 1) * self.audio_channels,
                                       self.num_sources * self.audio_channels,
                                       kernel_size=1, stride=1, padding=0, bias=False)

        if rescale:
            rescale_module(self, reference=rescale)

    @torch.cuda.amp.autocast(enabled=False)
    def _spec(self, x):
        x = x.float()
        hl = self.hop
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl, window=self.window.float())[..., :-1, :]

        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        z = torch.view_as_real(z)
        return z

    @torch.cuda.amp.autocast(enabled=False)
    def _ispec(self, z, length=None, scale=0):
        z = z.float()
        z = torch.view_as_complex(z.float().contiguous())
        hl = self.hop // (4 ** scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le, window=self.window.float(), skip_nola=False)
        x = x[..., pad: pad + length]
        return x

    def process_mix(self, x):

        # process mix
        pad_end = ((2 ** self.levels) * math.ceil(x.shape[-1] / (2 ** self.levels)) - x.shape[-1])
        xt = F.pad(x, (0, pad_end), mode='constant', value=0)

        b, c, t = xt.shape

        # spectrogram
        x_spec = self._spec(xt)  # b c f t2 2

        x_spec = rearrange(x_spec, 'b c f t2 r -> b (c r) f t2 ')  # b c (f=nfft//2) t2

        # normalize spectrogram
        mean_f = torch.mean(x_spec, dim=(1, 2, 3), keepdim=True)
        std_f = torch.std(x_spec, dim=(1, 2, 3), keepdim=True)
        freq_inputs = (x_spec - mean_f) / (std_f + 1e-5)

        # freq encoder
        freq_outputs, skips_f, lengths_f = self.f_encoder(freq_inputs)

        # normalize time inputs
        mean = torch.mean(xt, dim=(1, 2), keepdim=True)
        std = torch.std(xt, dim=(1, 2), keepdim=True)
        xt_normed = (xt - mean) / (std + 1e-5)

        # time encoder
        time_inputs = xt_normed
        time_outputs, skips_t, lengths_t = self.t_encoder(time_inputs)

        # wavelet transform
        with torch.cuda.amp.autocast(enabled=False):
            l_x, h_x = self.dwt(xt_normed.float())
        wavelet_inputs = [l_x] + h_x[::-1]

        wavelet_decoder_outputs = []
        # time_tsfmr_outputs = 0
        freq_tsfmr_outputs = 0
        for i in range(self.num_steps):
            # wavelet branch
            wavelet_outs, skips, lengths = self.wavelet_encoders[i](wavelet_inputs[i])

            # cross transformer
            freq_tsfmr_inp = self.freq_branch_compress[i](freq_outputs)

            freq_out, wavelet_out = self.cross_transformer[i](freq_tsfmr_inp, wavelet_outs)

            # wavelet decoder
            wavelet_decoder_outputs.append(self.wavelet_decoders[i](wavelet_out, skips, lengths))

            freq_tsfmr_outputs = freq_tsfmr_outputs + self.freq_branch_decompress[i](freq_out)

        freq_tsfmr_outputs, time_tsfmr_outputs = self.cross_transformer_full(freq_tsfmr_outputs, time_outputs)

        # inverse wavelet transform
        with torch.cuda.amp.autocast(enabled=False):
            out_dwt = [o.float() for o in wavelet_decoder_outputs]
            out_idwt = self.idwt([out_dwt[0], out_dwt[1:][::-1]])
        out_idwt = out_idwt * std + mean

        # freq decoder
        out_spec = self.f_decoder(freq_tsfmr_outputs, skips_f, lengths_f)
        # inverse stft
        out_spec = out_spec * std_f + mean_f
        out_spec = rearrange(out_spec, 'b (s c r) f t2 -> b (s c) f t2 r', r=2, s=self.num_sources)
        out_ispec = self._ispec(out_spec, length=t)

        # time decoder
        out_time = self.t_decoder(time_tsfmr_outputs, skips_t, lengths_t)
        out_time = out_time * std + mean

        y_hat = out_ispec + out_idwt + out_time

        y_hat = torch.cat([y_hat, xt], dim=1)
        y_hat = self.output_filter(y_hat)

        y_hat = rearrange(y_hat, 'b (s c) t -> b s c t', s=self.num_sources)

        if pad_end > 0:
            y_hat = y_hat[:, :, :, :-pad_end]

        return y_hat

    def forward(self, sources):
        if self.training:
            sources = self.augmenter(sources)
        x = sources.sum(dim=1)
        y = sources[:, self.target_indices]
        _y_hat = self.process_mix(x)
        output = {
            'y_hat': _y_hat,
        }

        targets = {
            'y': y,
        }

        return output, targets, sources
