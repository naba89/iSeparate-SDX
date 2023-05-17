"""Copyright: Nabarun Goswami (2023)."""
import pywt
import torch
import torch.nn as nn


class Wavelet(nn.Module):
    def __init__(self, wave):
        super().__init__()
        if wave == 'fk18':
            dec_lo = torch.tensor([0.00163579388754184,	-0.00467988492535553,	-0.000326360842339043,	0.0136393132491514,	-0.00615732925522970,	-0.0266324956435601,	0.0186954006791626,	0.0459366044858917,	-0.0397238678954921,	-0.0752406228087068,	0.0707029232219454,	0.127848450757352,	-0.113622551549222,	-0.246197980049399,	0.142345178925966,	0.650983106784119,	0.633556363915235,	0.221451519436035], dtype=torch.float32)
            dec_hi = torch.tensor([-0.221451519436035,	0.633556363915235,	-0.650983106784119,	0.142345178925966,	0.246197980049399,	-0.113622551549222,	-0.127848450757352,	0.0707029232219454,	0.0752406228087068,	-0.0397238678954921,	-0.0459366044858917,	0.0186954006791626,	0.0266324956435601,	-0.00615732925522970,	-0.0136393132491514,	-0.000326360842339043,	0.00467988492535553,	0.00163579388754184])
            rec_lo = torch.tensor([0.221451519436035,	0.633556363915235,	0.650983106784119,	0.142345178925966,	-0.246197980049399,	-0.113622551549222,	0.127848450757352,	0.0707029232219454,	-0.0752406228087068,	-0.0397238678954921,	0.0459366044858917,	0.0186954006791626,	-0.0266324956435601,	-0.00615732925522970,	0.0136393132491514,	-0.000326360842339043,	-0.00467988492535553,	0.00163579388754184], dtype=torch.float32)
            rec_hi = torch.tensor([0.00163579388754184,	0.00467988492535553,	-0.000326360842339043,	-0.0136393132491514, -0.00615732925522970,	0.0266324956435601,	0.0186954006791626,	-0.0459366044858917,	-0.0397238678954921,	0.0752406228087068,	0.0707029232219454,	-0.127848450757352,	-0.113622551549222,	0.246197980049399,	0.142345178925966,	-0.650983106784119,	0.633556363915235,	-0.221451519436035], dtype=torch.float32)
            filter_bank = torch.stack([dec_lo, dec_hi, rec_lo, rec_hi], dim=0)
            self.dec_len = 18
            self.rec_len = 18
        else:
            wavelet = pywt.Wavelet(wave)
            dec_lo = torch.tensor(wavelet.dec_lo, dtype=torch.float32)
            dec_hi = torch.tensor(wavelet.dec_hi, dtype=torch.float32)
            rec_lo = torch.tensor(wavelet.rec_lo, dtype=torch.float32)
            rec_hi = torch.tensor(wavelet.rec_hi, dtype=torch.float32)
            self.dec_len = wavelet.dec_len
            self.rec_len = wavelet.rec_len
            filter_bank = torch.stack([torch.tensor(f, dtype=torch.float32) for f in wavelet.filter_bank], dim=0)

        self.register_buffer('dec_lo', dec_lo)
        self.register_buffer('dec_hi', dec_hi)
        self.register_buffer('rec_lo', rec_lo)
        self.register_buffer('rec_hi', rec_hi)
        self.register_buffer('filter_bank', filter_bank)

    def forward(self, x):
        pass
