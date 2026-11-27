"""
fit_invivo_v3.py — In-vivo dataset v3 fitting driver.

Data:   data/invivo_Rx_200T_100f_3.xlsx
Tx:     Scalar baseline 0.052738597 (uniform across all frequencies)
Output: invivo_result.pkl
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
GPR_SRC = HERE / "gpr_models.py"
DATA_DIR = HERE / "data" if (HERE / "data").exists() else HERE.parent / "data"

# ── Load pipeline module via importlib (filename has a space) ─────────
_spec = importlib.util.spec_from_file_location("gpr_src", GPR_SRC)
gpr = importlib.util.module_from_spec(_spec)
sys.modules["gpr_src"] = gpr
_spec.loader.exec_module(gpr)

_main_mod = sys.modules["__main__"]
for _name in dir(gpr):
    if _name.startswith("__"):
        continue
    _obj = getattr(gpr, _name)
    if not hasattr(_main_mod, _name):
        setattr(_main_mod, _name, _obj)
    if getattr(_obj, "__module__", None) == "gpr_src":
        try:
            _obj.__module__ = "__main__"
        except (AttributeError, TypeError):
            pass


# ── Heart rate estimation ─────────────────────────────────────────────
def estimate_heart_rate(Ra, Ta, T_STEP, freq_range=(0.5, 4.0)):
    """Estimate heart rate from FFT of mag_dB along time axis."""
    Ta_col = Ta[:, np.newaxis]
    ratio = Ra / np.where(Ta_col > 1e-15, Ta_col, 1e-15)
    mag_dB = 20.0 * np.log10(np.where(ratio > 1e-15, ratio, 1e-15))

    N_time = mag_dB.shape[1]
    fft_freqs = np.fft.rfftfreq(N_time, d=T_STEP)
    mask = (fft_freqs >= freq_range[0]) & (fft_freqs <= freq_range[1])

    power_sum = np.zeros(mask.sum())
    for i in range(mag_dB.shape[0]):
        sig = mag_dB[i] - mag_dB[i].mean()
        spectrum = np.abs(np.fft.rfft(sig))
        power_sum += spectrum[mask]

    peak_idx = np.argmax(power_sum)
    hr_hz = fft_freqs[mask][peak_idx]
    print(f"  [Heart Rate] Estimated: {hr_hz:.3f} Hz ({hr_hz*60:.1f} bpm)")
    return hr_hz


# ── Config ────────────────────────────────────────────────────────────
T_STEP      = 0.05
N_RESTARTS  = 5    # [revision] 1 -> 5 (R3-7 asks for 5-10; 5 chosen to bound
                   #            wall-clock, maxiter now 200 instead of 50)
N_HOLDOUT_F = 16
N_HOLDOUT_T = 16
SAVE_PREFIX = "invivo"
TX_BASELINE = 0.052738597
OUT_PKL     = HERE / f"{SAVE_PREFIX}_result.pkl"

print("=" * 60)
print("  In-vivo (dataset v3) fitting")
print("  Source:  data/invivo_Rx_200T_100f_3.xlsx")
print(f"  Tx baseline: {TX_BASELINE}")
print(f"  Output:  {OUT_PKL.name}")
print("=" * 60)

# ── Load frequency axis ──────────────────────────────────────────────
freqs = gpr.load_freqs_from_file(os.path.join(DATA_DIR, "f_LOG100.xlsx"))
N_FREQ = len(freqs)

# ── Load Rx data and construct Tx ────────────────────────────────────
times_full = np.arange(0.0, 10.0, T_STEP)          # 200 points
t_start_idx = int(5.0 / T_STEP)                    # 100
t_end_idx   = int(9.0 / T_STEP)                    # 180
times_in = times_full[t_start_idx:t_end_idx]        # 80 points, 5.00~8.95s

Ra_raw = pd.read_excel(DATA_DIR / "invivo_Rx_200T_100f_3.xlsx", header=None).values
assert Ra_raw.shape == (200, N_FREQ), \
    f"Ra shape mismatch: {Ra_raw.shape}, expected (200, {N_FREQ})"
Ra_full = Ra_raw.T                                  # (N_freq, 200)
Ra = Ra_full[:, t_start_idx:t_end_idx]              # (N_freq, 80)
Ta = np.full(N_FREQ, TX_BASELINE)

print(f"[In-vivo] Time window: {times_in[0]:.2f}~{times_in[-1]:.2f}s "
      f"(idx {t_start_idx}:{t_end_idx}), Ra shape: {Ra.shape}")

# ── Estimate heart rate ──────────────────────────────────────────────
heart_rate = estimate_heart_rate(Ra, Ta, T_STEP)

# ── Run pipeline ─────────────────────────────────────────────────────
result = gpr.run_pipeline(
    freqs, Ra, Ta, times_in,
    heart_rate=heart_rate, n_restarts=N_RESTARTS,
    n_holdout_f=N_HOLDOUT_F, n_holdout_t=N_HOLDOUT_T,
    save_prefix=SAVE_PREFIX, label="In-vivo",
)

if not OUT_PKL.exists():
    with open(OUT_PKL, "wb") as f:
        pickle.dump(result, f)
    print(f"[Save] {OUT_PKL}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  In-vivo (v3) metrics")
print("=" * 60)
print(f"  Estimated heart rate: {heart_rate:.3f} Hz ({heart_rate*60:.1f} bpm)")
print(f"{'Method':<14} {'RMSE (dB)':>12} {'R²':>10} {'CI95 (%)':>10}")
print("-" * 60)
for row in result["compare_results"]:
    print(f"{row['label']:<14} {row['rmse']:>12.4f} "
          f"{row['r2']:>10.4f} {row['cov95']:>10.2f}")
print()
