"""gru_cv.py — forward-chaining rolling-origin evaluation of the GRU predictor.

[revision, GRU round] Replaces the single 60/20 split of train_gru.py
(which is kept unchanged to reproduce the submitted numbers).

Protocol (pre-specified, see GRU_REVISION_PLAN.md §2)
-----------------------------------------------------
* Full 10 s in vivo record: 100 frequencies x 200 frames, 50 ms frame interval.
* Fold with origin o: history = frames [o-60, o) (3 s), forecast = [o, o+20) (1 s).
* Development origins o = 60, 70, 80  (forecasts start 3.0 / 3.5 / 4.0 s and end
  before 5.0 s).  Used ONLY to select hyperparameters and the model variant.
* Test origins o = 100, 110, ..., 180 (forecasts start 5.0 ... 9.0 s).  Never
  used for any choice.  o = 160 reproduces the submitted split.
* Inside a fold, the GRU is trained on the first 50 history frames and early-
  stopped on an autoregressive rollout over the last 10 history frames; it is
  then retrained on all 60 frames for the selected number of epochs.
* Normalization statistics, the fundamental frequency, harmonic and AR
  coefficients all come from the fold's own history only.
"""
from __future__ import annotations

import json
import pathlib
import time
from math import comb  # noqa: F401  (kept for parity with the GPR module)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gru_model import ChannelGRU1D

HERE = pathlib.Path(__file__).resolve().parent
DATA = next((p for p in (HERE / "data", HERE.parent / "data", HERE.parent.parent / "data")
             if (p / "f_LOG100.xlsx").exists()), HERE / "data")  # works in the repo and in the supplementary package; _cand
OUT = HERE / "cv_results"

DT = 0.05
TX_BASELINE = 0.052738597
N_HIST, N_VAL_TAIL, N_FC = 60, 10, 20
DEV_ORIGINS = (60, 70, 80)
TEST_ORIGINS = tuple(range(100, 181, 10))
SUBMITTED_ORIGIN = 160
LR, MAX_EP, PATIENCE = 1e-3, 2000, 100


# ── data ──────────────────────────────────────────────────────────────
def load_invivo():
    freqs = pd.read_excel(DATA / "f_LOG100.xlsx", header=None).values.ravel()
    Ra = pd.read_excel(DATA / "invivo_Rx_200T_100f_3.xlsx", header=None).values.T
    mag = 20.0 * np.log10(np.maximum(Ra / TX_BASELINE, 1e-15))
    return freqs, mag.astype(float)                     # (100, 200)


def fundamental(hist):
    """Dominant modulation frequency [Hz] from history only (pooled power)."""
    Z = hist - hist.mean(1, keepdims=True)
    t = np.arange(hist.shape[1]) * DT
    fg = np.arange(0.8, 2.6, 0.001)
    c, s = np.cos(2 * np.pi * np.outer(fg, t)), np.sin(2 * np.pi * np.outer(fg, t))
    P = ((Z @ c.T) ** 2 + (Z @ s.T) ** 2).sum(0)
    return float(fg[int(np.argmax(P))])


# ── baselines (all use history only) ─────────────────────────────────
def harmonic_fit(hist, K, f0, n_out):
    """Per-frequency least-squares mean + K harmonics of f0.
    Returns (fitted history, extrapolated forecast)."""
    t_h = np.arange(hist.shape[1]) * DT
    t_f = (hist.shape[1] + np.arange(n_out)) * DT
    def design(t):
        cols = [np.ones_like(t)]
        for k in range(1, K + 1):
            cols += [np.cos(2 * np.pi * k * f0 * t), np.sin(2 * np.pi * k * f0 * t)]
        return np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design(t_h), hist.T, rcond=None)
    return (design(t_h) @ coef).T, (design(t_f) @ coef).T


def seasonal_naive(hist, f0, n_out):
    T0 = 1.0 / f0
    t_h = list(np.arange(hist.shape[1]) * DT)
    out = np.zeros((hist.shape[0], n_out))
    for i in range(hist.shape[0]):
        tt, vv = list(t_h), list(hist[i])
        for k in range(n_out):
            t = (hist.shape[1] + k) * DT
            v = np.interp(t - T0, tt, vv); out[i, k] = v
            tt.append(t); vv.append(v)
    return out


def ar_forecast(hist, p, n_out):
    """Per-frequency AR(p) with intercept, least squares, recursive rollout."""
    out = np.zeros((hist.shape[0], n_out))
    for i, y in enumerate(hist):
        X = np.column_stack([np.ones(len(y) - p)] +
                            [y[p - j - 1:len(y) - j - 1] for j in range(p)])
        beta, *_ = np.linalg.lstsq(X, y[p:], rcond=None)
        buf = list(y[-p:])
        for k in range(n_out):
            v = beta[0] + sum(beta[j + 1] * buf[-j - 1] for j in range(p))
            out[i, k] = v; buf.append(v)
    return out


def baselines(hist, n_out, K, p):
    f0 = fundamental(hist)
    _, harm = harmonic_fit(hist, K, f0, n_out)
    return {
        "Persistence": np.repeat(hist[:, -1:], n_out, 1),
        "Climatology": np.repeat(hist.mean(1, keepdims=True), n_out, 1),
        "Seasonal naive": seasonal_naive(hist, f0, n_out),
        f"Harmonic (K={K})": harm,
        f"AR({p})": ar_forecast(hist, p, n_out),
    }, f0


# ── GRU ──────────────────────────────────────────────────────────────
def _windows(Zn, W):
    X = np.stack([Zn[:, t:t + W] for t in range(Zn.shape[1] - W)], 1)   # (F, n, W)
    Y = np.stack([Zn[:, t + W] for t in range(Zn.shape[1] - W)], 1)     # (F, n)
    return (torch.tensor(X.reshape(-1, W, 1), dtype=torch.float32),
            torch.tensor(Y.reshape(-1, 1), dtype=torch.float32))


@torch.no_grad()
def rollout(model, seed_norm, n_steps):
    """Batched autoregressive rollout. seed_norm: (F, W) -> (F, n_steps)."""
    model.eval()
    buf = torch.tensor(seed_norm[:, :, None], dtype=torch.float32)
    outs = []
    for _ in range(n_steps):
        y = model(buf)                                   # (F, 1)
        outs.append(y[:, 0])
        buf = torch.cat([buf[:, 1:, :], y[:, None, :]], 1)
    return torch.stack(outs, 1).numpy()


def _fit(series, W, H, seed, n_epochs=None, val=None):
    """Train ChannelGRU1D on normalized series (F, n).  If val=(seed_norm,
    target_norm) is given, early-stop on rollout RMSE and return best epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChannelGRU1D(hidden_size=H, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    X, Y = _windows(series, W)
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    epochs = n_epochs or MAX_EP
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        loss = lossf(model(X), Y); loss.backward(); opt.step()
        if val is not None:
            pv = rollout(model, val[0], val[1].shape[1])
            v = float(np.sqrt(np.mean((pv - val[1]) ** 2)))
            if v < best - 1e-6:
                best, best_ep, wait = v, ep, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
    if val is not None:
        model.load_state_dict(best_state)
        return model, best_ep, best
    return model, epochs, None


def gru_forecast(hist, W, H, seed, residual_K=None):
    """Forecast n_FC frames from a 60-frame history.

    residual_K=None : plain GRU on the normalized series.
    residual_K=K    : harmonic template (mean + K harmonics, f0 from history)
                      + GRU on the template residual.
    Returns forecast (F, N_FC) and a dict of diagnostics.
    """
    n_tr = N_HIST - N_VAL_TAIL
    t0 = time.perf_counter()

    def prep(block, n_out):
        if residual_K is None:
            base_h, base_f = np.zeros_like(block), np.zeros((block.shape[0], n_out))
        else:
            f0 = fundamental(block)
            base_h, base_f = harmonic_fit(block, residual_K, f0, n_out)
        r = block - base_h
        mu, sd = r.mean(1, keepdims=True), r.std(1, keepdims=True) + 1e-8
        return (r - mu) / sd, mu, sd, base_f

    # 1) early stopping on the last 10 history frames
    Zs, mu, sd, base_val = prep(hist[:, :n_tr], N_VAL_TAIL)
    tgt = (hist[:, n_tr:] - base_val - mu) / sd
    _, best_ep, best_val = _fit(Zs, W, H, seed, val=(Zs[:, -W:], tgt))
    # 2) refit on all 60 frames for the selected number of epochs
    Zf, mu, sd, base_fc = prep(hist, N_FC)
    model, _, _ = _fit(Zf, W, H, seed, n_epochs=max(best_ep, 1))
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    pz = rollout(model, Zf[:, -W:], N_FC)
    t_inf = time.perf_counter() - t1
    fc = pz * sd + mu + base_fc
    n_par = sum(p.numel() for p in model.parameters())
    return fc, dict(best_epoch=best_ep, val_rmse_norm=best_val,
                    train_s=t_train, infer_s=t_inf, n_params=n_par)


CKPTS = (250, 500, 1000, 2000)


def gru_forecasts(hist, W, H, seed, residual_K=None, ckpts=CKPTS, with_es=True):
    """All training regimes for one (fold, W, H, seed, variant).

    Returns {regime: forecast (F, N_FC)}, regime in {"ES"} U {"E250", ...}.
    "E<n>"  : trained on all 60 history frames for n full-batch epochs
              (one run, forecasts taken at the checkpoints).
    "ES"    : early-stopped on the last-10-frame rollout, then refit (see
              gru_forecast).
    The regime is a hyperparameter chosen on development folds only.
    """
    out, diag = {}, {}
    if with_es:
        out["ES"], diag["ES"] = gru_forecast(hist, W, H, seed, residual_K)
    t0 = time.perf_counter()
    if residual_K is None:
        base_h, base_f = np.zeros_like(hist), np.zeros((hist.shape[0], N_FC))
    else:
        base_h, base_f = harmonic_fit(hist, residual_K, fundamental(hist), N_FC)
    r = hist - base_h
    mu, sd = r.mean(1, keepdims=True), r.std(1, keepdims=True) + 1e-8
    Z = (r - mu) / sd
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChannelGRU1D(hidden_size=H, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    X, Y = _windows(Z, W)
    for ep in range(1, max(ckpts) + 1):
        model.train(); opt.zero_grad()
        loss = lossf(model(X), Y); loss.backward(); opt.step()
        if ep in ckpts:
            t1 = time.perf_counter()
            out[f"E{ep}"] = rollout(model, Z[:, -W:], N_FC) * sd + mu + base_f
            diag[f"E{ep}"] = dict(train_s=t1 - t0, infer_s=time.perf_counter() - t1,
                                  final_loss=float(loss.item()))
    return out, diag


# ── metrics ──────────────────────────────────────────────────────────
def metrics(pred, true):
    e = pred - true
    dem = lambda x: x - x.mean(1, keepdims=True)
    ss = (dem(true) ** 2).sum()
    return dict(rmse=float(np.sqrt((e ** 2).mean())),
                step_rmse=np.sqrt((e ** 2).mean(0)).tolist(),
                r2_dem=float(1 - ((dem(pred) - dem(true)) ** 2).sum() / ss))


def fold(mag, o):
    return mag[:, o - N_HIST:o], mag[:, o:o + N_FC]
