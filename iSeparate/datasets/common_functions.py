"""Copyright: Nabarun Goswami (2023)."""
import json
import os
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchdata.dataloader2 import (
    MultiProcessingReadingService,
    DataLoader2)
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm


def volume(x: torch.Tensor, floor=1e-8, sr=44100):
    """
    Return the volume in dBFS.
    """
    pooled = F.avg_pool1d(x ** 2, 1 * sr, padding=int(0.5*sr))
    pooled = pooled.mean(dim=0)
    return torch.log10(floor + pooled) * 10


def silence_filter(inputs, sr):
    """
    This function filters out the stems which are silent.
    Use this if the datapipe returns each source as a separate tensor.
    :param inputs:
    :param sr:
    :return:
    """
    silent_segments = volume(inputs[0], sr=sr) < -40
    non_zero = torch.count_nonzero(silent_segments)
    if non_zero / silent_segments.shape[-1] > 0.7:
        return False
    else:
        return True


def silence_filter2(inputs, sr):
    """
    This function filters out the songs that have more than half of the sources silent.
    Use this if the datapipe returns all sources at once as a single tensor.
    :param inputs:
    :param sr:
    :return:
    """
    audios, *aug = inputs
    s, c, t = audios.shape
    num_silent_src = 0
    for i in range(s):
        silent_segments = volume(audios[i], sr=sr) < -40
        non_zero = torch.count_nonzero(silent_segments)
        if non_zero / silent_segments.shape[-1] > 0.7:
            num_silent_src += 1
    if num_silent_src >= s / 2:
        return False
    else:
        return True


def make_path(song_dir):
    return Path(song_dir)


def new_song_folder(song_folder, metadata):
    if os.fspath(song_folder) in metadata:
        return False
    else:
        return True


def is_instrument_available(inputs, instr):
    song_folder = Path(inputs[0])
    instr_file = get_audio_file_with_proper_ext(song_folder / f'{instr}.wav')
    if instr_file:
        return True
    else:
        return False


def get_audio_file_with_proper_ext(filename):
    acceptable_exts = ['.wav', '.flac', '.mp3', '.ogg']
    if filename.exists():
        return filename
    else:
        for ext in acceptable_exts:
            if (filename.with_suffix(ext)).exists():
                return filename.with_suffix(ext)
        return False


def fix_lengths(audios, minimum=True):
    """
    Center trim to minimum length if minimum is True, else left pad to maximum length.
    :param audios: list of audios [(c, t), (c, t), ...]
    :return:
    """
    lengths = [audio.shape[-1] for audio in audios]
    if minimum:
        min_length = min(lengths)
        audios = [audio[:, :min_length] for audio in audios]
    else:
        max_length = max(lengths)
        audios = [F.pad(audio, (0, max_length - audio.shape[-1])) for audio in audios]
    return audios


def get_metadata(song_folder, mixing_sources, ext='wav', sample_rate=44100):
    mixture_file = get_audio_file_with_proper_ext(song_folder / f'mixture.{ext}')
    sr = sample_rate
    if mixture_file:
        mix, sr = torchaudio.load(os.fspath(mixture_file))
        resample_fn = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        mix = resample_fn(mix)
    else:
        audio_files = [song_folder / f'{s}.{ext}' for s in mixing_sources]
        audios = []
        for audio_file in audio_files:
            audio_file = get_audio_file_with_proper_ext(audio_file)
            if audio_file:
                audio, sr = torchaudio.load(os.fspath(audio_file))
                resample_fn = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                audios.append(resample_fn(audio))
        if len(audios) == 0:
            raise ValueError(f'No source audio files found in {song_folder}')
        audios = fix_lengths(audios)
        mix = sum(audios)

    mix_mono = mix.mean(0)
    length = mix_mono.shape[-1]
    mean = mix_mono.mean()
    std = mix_mono.std() + torch.finfo(torch.float32).eps

    return os.fspath(song_folder), sr, length, mean.item(), std.item()


def dump_metadata(data_roots, metadata_file, mixing_sources, ext='wav', sample_rate=44100, num_workers=4):
    songs = []
    for data_root in data_roots:
        songs += [os.path.join(data_root, f) for f in os.listdir(data_root)]
    songs = [d for d in songs if os.path.isdir(d) and '.ipynb_checkpoints' not in d]

    song_dp = IterableWrapper(songs).map(make_path)
    song_dp = song_dp.sharding_filter()

    metadata_file = Path(metadata_file)
    metadata = {}
    # if metadata_file.exists(): just update the metadata
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    song_dp = song_dp.filter(partial(new_song_folder, metadata=metadata))

    song_dp = song_dp.map(partial(get_metadata, mixing_sources=mixing_sources, ext=ext, sample_rate=sample_rate))

    mp_rs = MultiProcessingReadingService(num_workers=num_workers)
    song_dl = DataLoader2(song_dp, reading_service=mp_rs)

    for meta in tqdm(song_dl, desc=f'Adding new songs to {metadata_file}'):
        song_folder, sr, length, mean, std = meta
        metadata[song_folder] = {
            'sr': sr,
            'length': length,
            'mean': mean,
            'std': std
        }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)


def load_audios(inputs, ext, mixing_sources, sample_rate, mono, segment_length,
                noisy, normalize, random_segments, center_segments, return_song_name):
    song_folder, metadata = inputs
    song_folder = Path(song_folder)
    song_name = song_folder.stem
    orig_sr = metadata['sr']
    length = metadata['length']
    mean = metadata['mean']
    std = metadata['std']

    segment_length = int(segment_length * orig_sr) if segment_length is not None else None

    audio_files = [song_folder / f'{s}.{ext}' for s in mixing_sources]
    audios = []

    is_noisy = False
    noisy_count = 0
    for audio_file in audio_files:
        if noisy is not None:
            for name in noisy:
                if name in os.fspath(audio_file):
                    noisy_count += 1
        audio_file = get_audio_file_with_proper_ext(audio_file)
        if not audio_file:
            nc = 1 if mono else 2
            audio = torch.zeros(nc, 1)
            noisy_count -= 1
        else:
            resample_fn = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sample_rate)

            if segment_length is not None and segment_length > 0:
                if segment_length > length:
                    start = 0
                    num_frames = length
                    pad_end = segment_length - length

                else:
                    if random_segments:
                        start = random.randint(0, length - segment_length)
                    elif center_segments:
                        start = (length - segment_length) // 2
                    else:
                        start = 0
                    num_frames = segment_length
                    pad_end = 0
            else:
                start = 0
                num_frames = length
                pad_end = 0

            audio, _ = torchaudio.load(os.fspath(audio_file), frame_offset=start,
                                       num_frames=num_frames)
            audio = F.pad(audio, (0, pad_end))

            audio = resample_fn(audio)

        audios.append(audio)

    audios = pad_sequence([audio.transpose(0, 1) for audio in audios], batch_first=True)  # (S, T, C)
    audios = audios.transpose(1, 2)  # (S, C, T)

    if mono:
        audios = audios.mean(1, keepdim=True)
    else:
        if audios.shape[1] == 1:
            audios = audios.repeat(1, 2, 1)
        else:
            audios = audios[:, :2]

    if normalize:
        audios = (audios - mean) / std

    if noisy_count > len(audio_files) / 2:
        is_noisy = True

    if return_song_name:
        return audios, song_name, is_noisy
    else:
        return audios, is_noisy


def load_single_audio(inputs, instr, sample_rate, mono, segment_length,
                      noisy, normalize, random_segments, center_segments, return_song_name):
    song_folder, metadata = inputs
    song_folder = Path(song_folder)
    song_name = song_folder.stem
    orig_sr = metadata['sr']
    length = metadata['length']
    mean = metadata['mean']
    std = metadata['std']

    segment_length = int(segment_length * orig_sr) if segment_length is not None else None

    audio_file = get_audio_file_with_proper_ext(song_folder / f'{instr}.wav')
    assert audio_file, f'No audio file found for {instr} in {song_folder}'

    is_noisy = False
    if noisy is not None:
        for name in noisy:
            if name in os.fspath(audio_file):
                is_noisy = True

    resample_fn = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sample_rate)

    if segment_length is not None and segment_length > 0:
        if segment_length > length:
            start = 0
            num_frames = length
            pad_end = segment_length - length

        else:
            if random_segments:
                start = random.randint(0, length - segment_length)
            elif center_segments:
                start = (length - segment_length) // 2
            else:
                start = 0
            num_frames = segment_length
            pad_end = 0
    else:
        start = 0
        num_frames = length
        pad_end = 0

    audio, _ = torchaudio.load(os.fspath(audio_file), frame_offset=start,
                               num_frames=num_frames)


    audio = F.pad(audio, (0, pad_end))
    audio = resample_fn(audio)

    if mono:
        audio = audio.mean(0, keepdim=True)
    else:
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        else:
            audio = audio[:2]

    if normalize:
        audio = (audio - mean) / std

    if return_song_name:
        return audio, song_name, is_noisy
    else:
        return audio, is_noisy
