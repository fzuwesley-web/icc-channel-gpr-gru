"""Per-tone receiver SNR from a raw NI-HWS (HDF5) record at 100 MS/s.

Usage: python snr_from_raw.py <raw_file> [f_LOG100.xlsx]
The record is split into 1 ms excitation periods (100 000 samples). Every tone
is an integer multiple of 1 kHz and falls on one FFT bin. SNR of a tone =
mean tone power / mean power of the non-excited bins within +-25 kHz, per
1 ms period.
The file is read in blocks of 50 periods, so memory use stays small.
"""
import sys
import h5py
import numpy as np
import pandas as pd

raw = sys.argv[1]
grid = sys.argv[2] if len(sys.argv) > 2 else "../data/f_LOG100.xlsx"
fk = pd.read_excel(grid, header=None).to_numpy(float).ravel()
kb = np.rint(fk / 1e3).astype(int)
Np = 100_000
with h5py.File(raw, "r") as h:
    ds = h["wfm_group0/axes/axis1/data_vector/data"]
    sc = h["wfm_group0/axes/axis1/scale_coef"][()]
    nper = ds.shape[0] // Np // 50 * 50
    P = np.zeros(Np // 2 + 1)
    for b in range(0, nper, 50):
        blk = ds[b * Np:(b + 50) * Np].astype(np.float32) * sc[1] + sc[0]
        P += (np.abs(np.fft.rfft(blk.reshape(50, Np), axis=1)) ** 2).sum(0)
P /= nper
tones = set(kb.tolist())
snr = np.array([10 * np.log10(P[k] / P[[j for j in range(k - 25, k + 26) if j not in tones and j > 0]].mean()) for k in kb])
out = pd.DataFrame({"freq_kHz": kb, "SNR_per_1ms_dB": snr.round(1)})
out.to_csv("invivo_snr_per_tone.csv", index=False)
print(out.describe().round(1))
