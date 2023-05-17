"""Copyright: Nabarun Goswami (2023).
Part of this code are adapted from https://github.com/facebookresearch/demucs/
Original license at https://github.com/facebookresearch/demucs/blob/main/LICENSE
"""
import random

import augment
import numpy as np

import torch
import torch.nn as nn
import torchaudio


class Augmenter(nn.Module):
    """
    Class to combine and apply data augmentations
    """

    def __init__(self, augmentations):
        """
        initialize the required augmentations
        :param augmentation_list: list of strings with the augmentation names, and any parameters
        E.g. [['SwapChannel', {}], ['RandomGain', {}], ['RandomSignFlip', {}]]
        """
        super().__init__()
        self.augmentations = []
        if augmentations is not None:
            for k, v in augmentations.items():
                self.augmentations.append(globals()[k](**v))

        self.augmentations = nn.ModuleList(self.augmentations)

    def forward(self, audio):
        """
        :param audio: audio to be augmented, shape CxT and other flags and stuff
        :return: augmented audio of same shape as input
        """
        for aug_fn in self.augmentations:
            audio = aug_fn(audio)

        return audio


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """

    def __init__(self, proba=1, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = torch.argsort(torch.rand(groups, group_size, streams, 1, 1, device=device),
                                         dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class RandomGain(nn.Module):
    def __init__(self, low=0.25, high=1.25, p=0.5):
        super().__init__()
        self.low = low
        self.high = high
        self.p = p

    def forward(self, audio):
        batch, streams, channels, time = audio.size()
        device = audio.device
        if random.random() < self.p:
            scales = torch.empty(batch, streams, 1, 1, device=device).uniform_(self.low, self.high)
            audio = audio * scales
        return audio


class SwapChannel(nn.Module):
    def forward(self, audio):
        batch, streams, channels, time = audio.size()
        if channels == 2:
            left = torch.randint(2, (batch, streams, 1, 1), device=audio.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            audio = torch.cat([audio.gather(2, left), audio.gather(2, right)], dim=2)
        return audio


class RandomSignFlip(nn.Module):
    def forward(self, audio):
        batch, streams, channels, time = audio.size()
        device = audio.device
        signs = torch.randint(2, (batch, streams, 1, 1), device=device, dtype=torch.float32)
        audio = audio * (2 * signs - 1)
        return audio


class ZeroRandomSource(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, audio):
        batch, streams, channels, time = audio.size()
        if random.random() < self.p:
            source_to_zero = torch.randint(streams, (batch,))
            for b in range(batch):
                audio[b, source_to_zero[b], :, :] = 0
        return audio


class Delay(torch.nn.Module):
    def __init__(
        self,
        p=0.5,
        sample_rate=44100,
        volume_factor=0.5,
        min_delay=200,
        max_delay=500,
        delay_interval=50,
    ):
        super().__init__()
        self.p = p
        self.sample_rate = sample_rate
        self.volume_factor = volume_factor
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay_interval = delay_interval

    def calc_offsets(self, ms):
        return [int(m * (self.sample_rate / 1000)) for m in ms]

    def forward(self, audio):
        if random.random() > self.p:
            return audio
        batch, *rest_shape = audio.shape
        ms = random.choices(
            np.arange(self.min_delay, self.max_delay, self.delay_interval), k=batch
        )
        offsets = self.calc_offsets(ms)
        delayed_signal = torch.zeros_like(audio)
        for i, offset in enumerate(offsets):
            delayed_signal[i] = torch.roll(audio[i], offset, dims=-1)
            delayed_signal[i, ..., :offset] = 0

        delayed_signal = delayed_signal * self.volume_factor
        audio = (audio + delayed_signal) / 2
        return audio


class Reverse(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, audio):
        if random.random() < self.p:
            audio = torch.flip(audio, dims=[-1])
        return audio


class PitchShiftAndTimeStretch(nn.Module):
    def __init__(self, enabled=False):
        super().__init__()
        self.enabled = enabled
        self.p = 0.2
        self.max_pitch = 2
        self.max_tempo = 12
        self.tempo_std = 5
        self.sample_rate = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.n_freq = self.n_fft // 2 + 1
        self.register_buffer('window', torch.hann_window(self.n_fft))
        self.time_stretch = torchaudio.transforms.TimeStretch(hop_length=self.hop_length, n_freq=self.n_freq)

    def forward(self, waveform):
        """
        :CAUTION: Currently applies same random shift and stretch to all items in the batch,
        all sources and all channels, so it is disabled for now.
        TODO: figure out how to do it properly and efficiently,
        TODO: maybe a better option is to do it in the dataloader using CPU
        :param waveform:
        :return:
        """
        with torch.inference_mode():
            out_length = int((1 - 0.01 * self.max_tempo) * waveform.shape[-1]) if self.enabled else waveform.shape[-1]
            if random.random() > self.p:
                # pack batch
                shape = waveform.size()
                waveform = waveform.reshape(-1, shape[-1])

                delta_pitch = random.randint(-self.max_pitch, self.max_pitch)
                delta_tempo = random.gauss(0, self.tempo_std)
                delta_tempo = min(max(-self.max_tempo, delta_tempo), self.max_tempo)

                pitch_transform = torchaudio.transforms.PitchShift(sample_rate=self.sample_rate, n_steps=delta_pitch)
                for p in pitch_transform.parameters():
                    p.requires_grad = False
                waveform = pitch_transform(waveform)

                tempo = 1.0 + delta_tempo * 0.01
                spec_f = torch.stft(waveform,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.n_fft,
                                    window=self.window,
                                    center=True,
                                    pad_mode="reflect",
                                    normalized=False,
                                    onesided=True,
                                    return_complex=True,
                                    )
                spec_f = self.time_stretch(spec_f, overriding_rate=tempo)
                len_stretch = int(round(waveform.shape[-1] / tempo))
                waveform = torch.istft(spec_f,
                                       n_fft=self.n_fft,
                                       hop_length=self.hop_length,
                                       win_length=self.n_fft,
                                       window=self.window,
                                       length=len_stretch,
                                       )
                waveform = waveform.reshape(shape[:-1] + waveform.shape[-1:])

            waveform = waveform[..., :out_length]
        # print('time stretch', waveform.shape)
        return waveform


class Reverb(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        reverberance_min=0,
        reverberance_max=100,
        dumping_factor_min=0,
        dumping_factor_max=100,
        room_size_min=0,
        room_size_max=100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max
        self.src_info = {"rate": self.sample_rate}
        # self.target_info = {
        #     "channels": 1,
        #     "rate": self.sample_rate,
        # }

    def forward(self, audio):
        reverberance = torch.randint(
            self.reverberance_min, self.reverberance_max, size=(1,)
        ).item()
        dumping_factor = torch.randint(
            self.dumping_factor_min, self.dumping_factor_max, size=(1,)
        ).item()
        room_size = torch.randint(
            self.room_size_min, self.room_size_max, size=(1,)
        ).item()

        num_channels = audio.shape[0]
        effect_chain = (
            augment.EffectChain()
            .reverb(reverberance, dumping_factor, room_size)
            .channels(num_channels)
        )

        target_info = {'sample_rate': self.sample_rate, 'channels': num_channels, 'length': audio.shape[-1]}
        audio = effect_chain.apply(
            audio, src_info=self.src_info, target_info=target_info
        )
        # print('reverb', audio.shape)
        return audio


class PitchShiftAndTempoAndReverb:
    def __init__(
        self, sample_rate, pitch_shift_min=-2.0, pitch_shift_max=2.0,
            min_tempo=0.88, max_tempo=1.12,
            reverberance_min=0,
            reverberance_max=100,
            dumping_factor_min=0,
            dumping_factor_max=100,
            room_size_min=0,
            room_size_max=100,
    ):
        self.sample_rate = sample_rate
        self.pitch_shift_cents_min = int(pitch_shift_min * 100)
        self.pitch_shift_cents_max = int(pitch_shift_max * 100)
        self.src_info = {"rate": self.sample_rate}

        self.reverberance_min = reverberance_min
        self.reverberance_max = reverberance_max
        self.dumping_factor_min = dumping_factor_min
        self.dumping_factor_max = dumping_factor_max
        self.room_size_min = room_size_min
        self.room_size_max = room_size_max

        self.min_tempo = min_tempo
        self.max_tempo = max_tempo

    def process(self, x):
        n_steps = random.randint(self.pitch_shift_cents_min, self.pitch_shift_cents_max)
        reverberance = torch.randint(
            self.reverberance_min, self.reverberance_max, size=(1,)
        ).item()
        dumping_factor = torch.randint(
            self.dumping_factor_min, self.dumping_factor_max, size=(1,)
        ).item()
        room_size = torch.randint(
            self.room_size_min, self.room_size_max, size=(1,)
        ).item()
        delta_tempo = random.uniform(self.min_tempo, self.max_tempo)

        num_channels = x.shape[0]
        num_samples = x.shape[1]

        effect_chain = augment.EffectChain().pitch(n_steps).rate(self.sample_rate).\
            reverb(reverberance, dumping_factor, room_size).channels(num_channels) \
            .tempo(delta_tempo)
        num_channels = x.shape[0]
        target_info = {
            "channels": num_channels,
            "length": num_samples,
            "rate": self.sample_rate,
        }
        y = effect_chain.apply(x, src_info=self.src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        if y.shape[1] != x.shape[1]:
            if y.shape[1] > x.shape[1]:
                y = y[:, : x.shape[1]]
            else:
                y0 = torch.zeros(num_channels, x.shape[1]).to(y.device)
                y0[:, : y.shape[1]] = y
                y = y0
        return y

    def __call__(self, audio):
        if audio.ndim == 3:
            for b in range(audio.shape[0]):
                audio[b] = self.process(audio[b])
            return audio
        else:
            return self.process(audio)


def apply_augmenter(inputs, augmenter):
    audios, *rest = inputs
    return augmenter(audios), *rest
