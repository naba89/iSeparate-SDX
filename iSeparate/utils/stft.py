"""
This code is used to skip the NOLA check during istft to avoid CUDA synchronization
See https://github.com/pytorch/pytorch/issues/94718 for context
"""
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity


def istft(input_tensor, n_fft, hop_length=None, win_length=None, window=None,
          normalized=False, center=True, length=None, onesided=None, return_complex=False, skip_nola=True):
    r"""Invert a complex-valued STFT or spectrogram to a real-valued time-domain
            signal using an overlap-add procedure. But skips the NOLA constraint
            I don't handle the return complex case
    """
    if not skip_nola:
        y = torch.istft(input_tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                        normalized=normalized, center=center, length=length,
                        onesided=onesided, return_complex=return_complex)
    else:
        if return_complex:
            raise NotImplementedError("return_complex=True is not implemented")
        if hop_length is None:
            hop_length = int(n_fft // 4)
        if win_length is None:
            win_length = n_fft

        input_tensor = torch.view_as_real(input_tensor.resolve_conj())

        input_dim = input_tensor.dim()
        n_frames = input_tensor.size(-2)
        fft_size = input_tensor.size(-3)

        expected_output_signal_len = n_fft + hop_length * (n_frames - 1)

        onesided = onesided if onesided else fft_size != n_fft

        if window is None:
            window = torch.ones(n_fft, dtype=input_tensor.dtype, device=input_tensor.device)

        if win_length != n_fft:
            # center window by padding zeros on right and left side
            left = (n_fft - win_length) / 2
            window = F.pad(window, (left, n_fft - win_length - left), mode='constant', value=0)

        if input_dim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = torch.view_as_complex(input_tensor.transpose(1, 2))
        if not onesided:
            input_tensor = input_tensor[..., :fft_size // 2 + 1]

        norm_mode = 'ortho' if normalized else 'backward'

        input_tensor = torch.fft.irfft(input_tensor, dim=input_tensor.dim() - 1, norm=norm_mode)

        assert input_tensor.size(2) == n_fft

        y_tmp = input_tensor * window.view(1, 1, n_fft)

        fold_params = {
            'kernel_size': (n_fft, 1),
            'stride': (hop_length, 1),
        }

        # y_tmp = rearrange(y_tmp, 'b t f -> b f t')
        y_tmp = y_tmp.transpose(1, 2)
        y = F.fold(y_tmp, output_size=(expected_output_signal_len, 1), **fold_params)
        y = y.reshape(y.size(0), -1)
        window = window.pow(2).expand(1, n_frames, n_fft)

        # window = rearrange(window, 'b t f -> b f t')
        window = window.transpose(1, 2)

        window_envelop = F.fold(window, output_size=(expected_output_signal_len, 1), **fold_params)
        window_envelop = window_envelop.reshape(window_envelop.size(0), -1)
        assert window_envelop.size(1) == expected_output_signal_len
        assert y.size(1) == expected_output_signal_len

        start = n_fft // 2 if center else 0

        end = expected_output_signal_len
        if length is not None:
            end = start + length
        elif center:
            end = -(n_fft // 2)

        y = y[..., start:end]
        window_envelop = window_envelop[..., start:end]

        y = y / window_envelop

        if end > expected_output_signal_len:
            y = F.pad(y, (0, end - expected_output_signal_len), mode='constant', value=0)

    return y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_signal = torch.randn(256, 7 * 44100, device=device)

    # STFT
    spec_params = {
        'n_fft': 4096,
        'hop_length': 1024,
        'win_length': 4096,
        'window': torch.hann_window(4096, device=device),
        'normalized': True,
        'center': True,
        'onesided': True,
    }

    # sanity check
    dummy_X = torch.stft(dummy_signal, **spec_params, return_complex=True)
    dummy_x_hat = istft(dummy_X, **spec_params, length=dummy_signal.shape[-1], skip_nola=True)
    print(torch.allclose(dummy_signal, dummy_x_hat, atol=1e-6))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                 schedule=torch.profiler.schedule(
                     wait=5,
                     warmup=1,
                     active=1),
                 profile_memory=True,
                 with_stack=True,
                 record_shapes=True
                 ) as prof:
        for _ in range(8):
            dummy_spec = torch.stft(dummy_signal, **spec_params, return_complex=True)
            ispec_skip_nola = istft(dummy_spec, **spec_params, length=dummy_signal.shape[-1], skip_nola=True)
            ispec_nola = istft(dummy_spec, **spec_params, length=dummy_signal.shape[-1], skip_nola=False)
            prof.step()
