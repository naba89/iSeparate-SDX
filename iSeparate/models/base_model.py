"""Copyright: Nabarun Goswami (2023)."""
import random
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process_mix(self, mix):
        pass

    def separate(self, mix, patch_length=20, hop_length=10, sr=44100, use_window=False, disable_pbar=False, shifts=0, chunk_only_gpu=False):
        """
        This is a single source separation model.
        :param mix: input mixture of shape (batch_size, num_channels, num_samples)
        :param patch_length: length of the patches to be used for inference
        :param hop_length: overlap between patches
        :param sr: sampling rate
        :param use_window: hann_window is applied to the patches
        :param disable_pbar: disable progress bar
        :param shifts: number of random shifts to be applied to the mixture
        """

        if shifts > 0:
            max_shift = int(0.5 * sr)
            length = mix.shape[-1]
            mix = F.pad(mix, (max_shift, max_shift))
            out = 0
            for _ in range(shifts):
                offset = random.randint(0, max_shift)
                shifted = mix[..., offset:length + max_shift]
                shifted_out = self.separate(shifted, patch_length, hop_length, sr, use_window, disable_pbar, 0)
                out += shifted_out[..., max_shift - offset:]
            out /= shifts
            return out

        if patch_length is None:
            output_tensor = self.process_mix(mix)
        else:
            if chunk_only_gpu:
                mix = mix.cpu()

            patch_length = int(patch_length * sr)
            hop_length = int(hop_length * sr)

            length = mix.shape[-1]

            if use_window:
                weight = torch.hann_window(patch_length, requires_grad=False, device=mix.device)
            else:
                weight = hop_length / patch_length

            # Chunk the tensor along the last dimension
            chunks = torch.nn.functional.unfold(
                mix.unsqueeze(-1),
                kernel_size=(patch_length, 1),
                padding=(patch_length, 0),
                stride=(hop_length, 1),
            )

            # processing on the chunks here...
            nb_chunks = chunks.shape[-1]
            processed_chunks = []
            for i in tqdm(range(nb_chunks), leave=False, disable=disable_pbar):
                chunk = chunks[..., i].reshape(mix.shape[0], mix.shape[1], patch_length)
                if chunk_only_gpu:
                    chunk = chunk.cuda()
                processed_chunk = self.process_mix(chunk) * weight
                if chunk_only_gpu:
                    processed_chunk = processed_chunk.cpu()
                processed_chunks.append(processed_chunk)

            processed_chunks = torch.stack(processed_chunks, dim=-1)
            b, nsrc, c, l, n = processed_chunks.shape

            processed_chunks = processed_chunks.reshape(b * nsrc * c, l, n)

            # Overlap add the result
            overlap_added = torch.nn.functional.fold(
                processed_chunks,
                (length, 1),
                kernel_size=(patch_length, 1),
                padding=(patch_length, 0),
                stride=(hop_length, 1),
            )
            output_tensor = overlap_added.reshape(b, nsrc, c, length)

        return output_tensor

