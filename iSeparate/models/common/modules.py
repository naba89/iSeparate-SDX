"""Copyright: Nabarun Goswami (2023).
Part of this code is taken/modified from https://github.com/facebookresearch/demucs/tree/main
with the original license at https://github.com/facebookresearch/demucs/blob/main/LICENSE
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from iSeparate.models.common.utils import unfold


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).
    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2] ** 0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    """

    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class ResidualBlock(nn.Module):
    def __init__(self, channels, groups, compress, kernel_size, dilation, init,
                 lstm=False, attn=False, heads=4, ndecay=4):
        super().__init__()
        self.groups = groups
        hidden = channels // compress
        padding = dilation * (kernel_size // 2)

        mods = [("conv", nn.Conv1d(channels * groups, hidden * groups,
                                   kernel_size=kernel_size, groups=groups,
                                   stride=1, dilation=dilation,
                                   padding=padding))]
        if lstm:
            assert groups == 1, "LSTM not supported with groups"
            mods += [("lstm", BLSTM(hidden, layers=2, max_steps=200, skip=True))]
        elif attn:
            mods += [("attn", LocalState(hidden, heads=heads, ndecay=ndecay))]

        mods += [("ln", nn.GroupNorm(groups, hidden * groups)),
                 ("gelu", nn.GELU())]

        self.conv_ln_gelu = nn.Sequential(OrderedDict(mods))

        self.conv_ln = nn.Sequential(OrderedDict([("conv", nn.Conv1d(hidden * groups, 2 * channels * groups,
                                                                     kernel_size=1, groups=groups)),
                                                  ("ln", nn.GroupNorm(groups, 2 * channels * groups))]))
        self.glu = nn.GLU(1)

        self.layer_scale = LayerScale(channels, init=init)

    def forward(self, x):
        res = x
        x = self.conv_ln_gelu(x)
        x = self.conv_ln(x)
        x = rearrange(x, 'b (g c) t -> (b g) c t', g=self.groups)
        x = self.glu(x)
        x = self.layer_scale(x)
        x = rearrange(x, '(b g) c t -> b (g c) t', g=self.groups)
        return x + res


class ResidualBranch(nn.Module):
    def __init__(self, channels, groups, depth, compress, kernel_size, init, lstm=False, attn=False):
        super().__init__()
        self.residual_blocks = nn.ModuleList([ResidualBlock(channels, groups, compress,
                                                            kernel_size, 2 ** i, init, lstm, attn) for i in
                                              range(depth)])

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)
        return x


class EncLayer(nn.Module):
    def __init__(self, freq, in_channels, out_channels, kernel_size, stride, groups, context, residual_params,
                 norm=True, norm_groups=1, lstm=False, attn=False):
        super().__init__()
        self.groups = groups
        self.freq = freq
        self.stride = stride
        if stride == 4:
            pad = kernel_size // stride
        elif stride == 2:
            pad = int(kernel_size / stride - 1)
        else:
            raise ValueError("Stride must be 2 or 4")
        if freq:
            conv_class = nn.Conv2d
            kernel_size = (kernel_size, 1)
            stride = (stride, 1)
            pad = (pad, 0)
        else:
            conv_class = nn.Conv1d

        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups * groups, d)  # noqa

        self.conv_ln_gelu = nn.Sequential(OrderedDict([("conv", conv_class(in_channels * groups, out_channels * groups,
                                                                           kernel_size, stride, padding=pad,
                                                                           groups=groups)),
                                                       ("ln", norm_fn(out_channels * groups)),
                                                       ("gelu", nn.GELU())]))

        self.residual_branch = ResidualBranch(channels=out_channels, groups=groups,
                                              attn=attn, lstm=lstm, **residual_params)

        self.conv_ln = nn.Sequential(OrderedDict([("conv", conv_class(out_channels * groups, 2 * out_channels * groups,
                                                                      1 + 2 * context, 1, context,
                                                                      groups=groups)),
                                                  ("ln", norm_fn(2 * out_channels * groups))]))
        self.glu = nn.GLU(1)

    def forward(self, x):
        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))

        out = self.conv_ln_gelu(x)
        if self.freq:
            b, c, f, t = out.shape
            out = out.permute(0, 2, 1, 3).reshape(-1, c, t)
            out = self.residual_branch(out)
            out = out.view(b, f, c, t).permute(0, 2, 1, 3)
        else:
            out = self.residual_branch(out)

        out = self.conv_ln(out)
        out = rearrange(out, 'b (g c) ... -> (b g) c ...', g=self.groups)
        out = self.glu(out)
        out = rearrange(out, '(b g) c ... -> b (g c) ...', g=self.groups)

        return out


class DecLayer(nn.Module):
    def __init__(self, freq, in_channels, out_channels, kernel_size, stride, groups, last, context, residual_params,
                 norm=True, norm_groups=1, lstm=False, attn=False):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups
        self.freq = freq
        if stride == 4:
            self.pad = kernel_size // stride
        elif stride == 2:
            self.pad = int(kernel_size / stride - 1)
        else:
            raise ValueError("Stride must be 2 or 4")

        if last:
            act = nn.Identity()
        else:
            act = nn.GELU()

        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups * groups, d)  # noqa

        if freq:
            conv_tr_class = nn.ConvTranspose2d
            conv_class = nn.Conv2d
            kernel_size = (kernel_size, 1)
            stride = (stride, 1)
            # pad = (pad, 0)
        else:
            conv_tr_class = nn.ConvTranspose1d
            conv_class = nn.Conv1d

        self.conv_ln = nn.Sequential(OrderedDict([("conv", conv_class(in_channels * groups, 2 * in_channels * groups,
                                                                      1 + 2 * context, 1, context,
                                                                      groups=groups)),
                                                  ("ln", norm_fn(2 * in_channels * groups))]))
        self.glu = nn.GLU(1)

        self.residual_branch = ResidualBranch(channels=in_channels, groups=groups,
                                              attn=attn, lstm=lstm,
                                              **residual_params)

        self.tr_conv_ln_act = nn.Sequential(
            OrderedDict([("tr_conv", conv_tr_class(in_channels * groups,
                                                   out_channels * groups,
                                                   kernel_size, stride, groups=groups)),
                         ("ln", norm_fn(out_channels * groups)),
                         ("act", act)]))

    def forward(self, x, skip, length):
        if skip is not None:
            x = x + skip

        out = self.conv_ln(x)
        out = rearrange(out, 'b (g c) ... -> (b g) c ...', g=self.groups)
        out = self.glu(out)
        out = rearrange(out, '(b g) c ... -> b (g c) ...', g=self.groups)

        if self.freq:
            b, c, f, t = out.shape
            out = out.permute(0, 2, 1, 3).reshape(-1, c, t)
            out = self.residual_branch(out)
            out = out.view(b, f, c, t).permute(0, 2, 1, 3)
        else:
            out = self.residual_branch(out)

        out = self.tr_conv_ln_act(out)

        if self.freq:
            if self.pad > 0:
                out = out[..., self.pad:-self.pad, :]
        else:
            out = out[..., self.pad:self.pad + length]
        return out


class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth=True):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class Encoder(nn.Module):
    def __init__(self, freq, in_channels, enc_channels, hidden_size, encoder_params, groups=1, norm_starts=0,
                 lstm_starts=None, attn_starts=None,
                 freq_emb_weight=None, num_freqs=None):
        super().__init__()
        self.freq = freq
        enc_in_channels = in_channels
        self.enc_layers = nn.ModuleList()
        self.freq_emb = None

        for i, (enc_out_channels, stride) in enumerate(enc_channels):
            self.enc_layers.append(EncLayer(freq, enc_in_channels, enc_out_channels, stride=stride, groups=groups,
                                            norm=i >= norm_starts,
                                            lstm=i >= lstm_starts if lstm_starts is not None else False,
                                            attn=i >= attn_starts if attn_starts is not None else False,
                                            **encoder_params))
            enc_in_channels = enc_out_channels

            if freq and freq_emb_weight is not None and i == 0:
                self.freq_emb = ScaledEmbedding(
                    num_freqs // stride, enc_in_channels
                )
                self.freq_emb_scale = freq_emb_weight

        if hidden_size != enc_channels[-1][0]:
            if freq:
                self.final_conv = nn.Conv2d(enc_channels[-1][0] * groups, hidden_size * groups, 1, groups=groups)
            else:
                self.final_conv = nn.Conv1d(enc_channels[-1][0] * groups, hidden_size * groups, 1, groups=groups)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x):
        skips = []
        lengths = [x.shape[-1]]
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x)
            if i == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            skips.append(x)
            lengths.append(x.shape[-1])
        x = self.final_conv(x)
        return x, skips, lengths


class Decoder(nn.Module):
    def __init__(self, freq, enc_channels, out_channels, hidden_size, decoder_params, groups=1, norm_starts=0,
                 lstm_starts=None, attn_starts=None):
        super().__init__()
        self.freq = freq
        if hidden_size != enc_channels[-1][0]:
            if freq:
                self.init_conv = nn.Conv2d(hidden_size * groups, enc_channels[-1][0] * groups, 1, groups=groups)
            else:
                self.init_conv = nn.Conv1d(hidden_size * groups, enc_channels[-1][0] * groups, 1, groups=groups)
        else:
            self.init_conv = nn.Identity()

        dec_in_channels = enc_channels[-1][0]
        strides = [enc_channels[i][1] for i in range(len(enc_channels))][::-1]
        dec_out_channels = enc_channels[::-1][1:]
        self.dec_layers = nn.ModuleList()
        for i in range(len(dec_out_channels) + 1):
            if i == len(enc_channels) - 1:
                last = True
                dec_out_ch = out_channels
            else:
                dec_out_ch = dec_out_channels[i][0]
                last = False
            norm = i <= len(dec_out_channels) - norm_starts
            lstm = i <= len(dec_out_channels) - lstm_starts if lstm_starts is not None else False
            attn = i <= len(dec_out_channels) - attn_starts if attn_starts is not None else False
            self.dec_layers.append(DecLayer(freq, dec_in_channels, dec_out_ch, groups=groups,
                                            last=last, stride=strides[i], norm=norm,
                                            lstm=lstm, attn=attn,
                                            **decoder_params))
            dec_in_channels = dec_out_ch

    def forward(self, x, skips, lengths):
        x = self.init_conv(x)
        lengths = lengths[::-1][1:]
        skips = skips[::-1]
        for i, dec_layer in enumerate(self.dec_layers):
            x = dec_layer(x, skips[i], lengths[i])
        return x
