"""Copyright: Nabarun Goswami (2023)."""
import copy
import hashlib
import json
from functools import partial

import torch.distributed as dist
from torchaudio_augmentations import RandomApply, Compose
from torchdata.dataloader2 import (
    MultiProcessingReadingService, SequentialReadingService,
    DataLoader2, DistributedReadingService)
from torchdata.datapipes.iter import IterableWrapper, Zipper

from iSeparate.datasets.augmentation_modules import apply_augmenter, PitchShiftAndTempoAndReverb
from iSeparate.datasets.common_functions import load_audios, dump_metadata, silence_filter2, is_instrument_available, \
    load_single_audio, silence_filter


def stem_folder_dataloader(data_roots, metadata_file_prefix, mixing_sources, noisy,
                           sample_rate, mono, segment_length, normalize, center,
                           shuffle, random_segments, num_workers, batch_size, cycle,
                           drop_last, return_song_name, filter_silence, return_dp=False):
    """
    Create a dataloader for the stem folder dataset. Expects each song to have all sources present.
    Within batch random mixing can be done on-the-fly later on.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print(f'Using stem_folder_dataloader with mixing sources: {mixing_sources}')

    mixing_sources = sorted(mixing_sources)
    signature = hashlib.md5(json.dumps(data_roots).encode()).hexdigest()[:8]
    metadata_file = f'{metadata_file_prefix}_{signature}.json'
    if rank == 0:
        dump_metadata(data_roots, metadata_file, mixing_sources,
                      ext='wav', sample_rate=sample_rate, num_workers=num_workers)
    if world_size > 1:
        dist.barrier()

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    metadata = list(metadata.items())
    song_dp = IterableWrapper(metadata)
    if cycle:
        song_dp = song_dp.cycle()
    else:
        song_dp.set_length(len(metadata))
    if shuffle:
        song_dp = song_dp.shuffle()
    song_dp = song_dp.sharding_filter()

    song_dp = song_dp.map(partial(load_audios,
                                  ext='wav',
                                  mixing_sources=mixing_sources,
                                  sample_rate=sample_rate, mono=mono, noisy=noisy,
                                  normalize=normalize, segment_length=segment_length,
                                  random_segments=random_segments, center_segments=center,
                                  return_song_name=return_song_name))
    if filter_silence:
        song_dp = song_dp.filter(partial(silence_filter2, sr=sample_rate))

    song_dp = song_dp.batch(batch_size, drop_last=drop_last).collate()

    if return_dp:
        return song_dp

    if world_size > 1:
        dist_rs = DistributedReadingService()
        if num_workers > 0:
            mp_rs = MultiProcessingReadingService(num_workers=num_workers)
            rs = SequentialReadingService(dist_rs, mp_rs)
        else:
            rs = dist_rs
    else:
        if num_workers > 0:
            rs = MultiProcessingReadingService(num_workers=num_workers)
        else:
            rs = None

    song_dl = DataLoader2(song_dp, reading_service=rs)
    return song_dl


def random_mix_training_dataloader(data_roots, metadata_file_prefix, mixing_sources, noisy,
                                   sample_rate, mono, segment_length, normalize, center,
                                   shuffle, random_segments, num_workers, batch_size, cycle,
                                   drop_last, return_song_name, filter_silence, cpu_aug):
    """
    Create a dataloader for random mix training. Creates separate datapipes for each source and zips them together.
    This allows for song folders without all sources present and also
    allows for potentially different precessing for each source.

    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print(f'Using random_mix_training_dataloader with mixing sources: {mixing_sources}')

    mixing_sources = sorted(mixing_sources)
    signature = hashlib.md5(json.dumps(data_roots).encode()).hexdigest()[:8]
    metadata_file = f'{metadata_file_prefix}_{signature}.json'
    if rank == 0:
        dump_metadata(data_roots, metadata_file, mixing_sources,
                      ext='wav', sample_rate=sample_rate, num_workers=num_workers)
    if world_size > 1:
        dist.barrier()

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    song_dp = IterableWrapper(list(metadata.items()))

    # if shuffle:
    #     song_dp = song_dp.shuffle()

    instr_dps = [copy.deepcopy(song_dp) for _ in range(len(mixing_sources))]
    # instr_dps = song_dp.fork(len(mixing_sources))

    for i, instr in enumerate(mixing_sources):
        instr_dps[i] = instr_dps[i].filter(partial(is_instrument_available, instr=instr))

        if cycle:
            instr_dps[i] = instr_dps[i].cycle()
        if shuffle:
            instr_dps[i] = instr_dps[i].shuffle()

        instr_dps[i] = instr_dps[i].sharding_filter()

        instr_dps[i] = instr_dps[i].map(partial(load_single_audio,
                                                instr=instr,
                                                sample_rate=sample_rate,
                                                mono=mono,
                                                segment_length=segment_length,
                                                noisy=noisy,
                                                normalize=normalize,
                                                random_segments=random_segments,
                                                center_segments=center,
                                                return_song_name=return_song_name))
        if filter_silence:
            instr_dps[i] = instr_dps[i].filter(partial(silence_filter, sr=sample_rate))

        if cpu_aug:
            transforms = [
                # PitchShiftAndTimeStretch(enabled=True),
                RandomApply([PitchShiftAndTempoAndReverb(sample_rate=sample_rate)], p=0.2)
            ]
            transform = Compose(transforms=transforms)
            instr_dps[i] = instr_dps[i].map(partial(apply_augmenter, augmenter=transform))

    training_dp = Zipper(*instr_dps).collate()

    training_dp = training_dp.batch(batch_size, drop_last=drop_last).collate()

    if world_size > 1:
        dist_rs = DistributedReadingService()
        if num_workers > 0:
            mp_rs = MultiProcessingReadingService(num_workers=num_workers)
            rs = SequentialReadingService(dist_rs, mp_rs)
        else:
            rs = dist_rs
    else:
        if num_workers > 0:
            rs = MultiProcessingReadingService(num_workers=num_workers)
        else:
            rs = None

    song_dl = DataLoader2(training_dp, reading_service=rs)
    return song_dl
