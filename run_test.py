"""run_test.py — evaluation on TEST folds with the configuration locked by run_dev.py.

* Both GRU variants at their dev-selected (H, W, regime), seeds 0-4, on test
  folds (and on dev folds, used ONLY to calibrate the ensemble interval scale).
* Baselines on test folds: persistence, climatology, seasonal naive,
  harmonic(K*), AR(p*), and PF 1D GPR extrapolation (same kernel as Sec. IV-B).
* Ensemble: mean of 5 seeds; interval = mean +- z * lambda * spread, with lambda
  chosen on dev folds so that the 95 % interval covers 95 % there.
* Fold-level bootstrap (2000 resamples) for step-wise RMSE CIs, and the
  look-ahead windows h_skill (vs climatology) and h_adv (vs the best simple
  baseline on dev), defined as the largest h such that the 95 % CI of the
  RMSE difference lies entirely below zero for every step 1..h.
"""
import importlib.util
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, ".")
import gru_cv as G  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)
NBOOT = 2000


def _gru_task(args):
    import torch
    torch.set_num_threads(1)
    vname, K, H, W, regime, o, seed = args
    _, mag = G.load_invivo()
    hist, _ = G.fold(mag, o)
    t0 = time.perf_counter()
    if regime == "ES":
        fc, d = G.gru_forecast(hist, W, H, seed, residual_K=K)
        tr, inf = d["train_s"], d["infer_s"]
    else:
        E = int(regime[1:])
        fcs, d = G.gru_forecasts(hist, W, H, seed, residual_K=K, ckpts=(E,), with_es=False)
        fc, tr, inf = fcs[regime], d[regime]["train_s"], d[regime]["infer_s"]
    return vname, o, seed, fc, tr, inf


def pf_gpr(freqs, hist):
    spec = importlib.util.spec_from_file_location(
        "gprmod", str(pathlib.Path(__file__).resolve().parent.parent / "gpr" / "gpr_models.py"))
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    t_h = np.arange(hist.shape[1]) * G.DT
    t_f = (hist.shape[1] + np.arange(G.N_FC)) * G.DT
    mfn = g.FrequencyMeanFunction(3).fit(freqs, hist)
    m = g.PerFrequencyGPR1D(freqs, t_h, heart_rate_hz=G.fundamental(hist), n_restarts=5)
    t0 = time.perf_counter()
    m.fit(mfn.residual(freqs, hist))
    mp_, ms_ = m.predict(t_f)
    return mfn.restore(freqs, mp_), ms_, time.perf_counter() - t0


def lookahead(diff_boot):
    """diff_boot: (NBOOT, 20) bootstrap mean RMSE differences (method - ref)."""
    upper = np.percentile(diff_boot, 97.5, axis=0)
    h = 0
    for u in upper:
        if u < 0: h += 1
        else: break
    return h


def main():
    sel = json.load(open(G.OUT / "dev_selection.json"))
    K, p = sel["K_star"], sel["p_star"]
    freqs, mag = G.load_invivo()
    origins_all = tuple(G.DEV_ORIGINS) + tuple(G.TEST_ORIGINS)
    variants = {}
    for vname, cfg in sel["gru"].items():
        variants[vname] = (None if vname == "GRU" else K, cfg["H"], cfg["W"], cfg["regime"])
    tasks = [(v, c[0], c[1], c[2], c[3], o, s) for v, c in variants.items()
             for o in origins_all for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(6) as pool:
        res = list(pool.imap_unordered(_gru_task, tasks))
    print(f"GRU runs: {len(res)} in {time.time() - t0:.0f}s", flush=True)

    fc = {v: {o: [None] * len(SEEDS) for o in origins_all} for v in variants}
    times = {v: [] for v in variants}
    for v, o, s, f, tr, inf in res:
        fc[v][o][SEEDS.index(s)] = f; times[v].append((tr, inf))

    # ensemble calibration on dev folds
    lam = {}
    for v in variants:
        best_l, best_gap = 1.0, 9.0
        for l in np.arange(0.5, 30.01, 0.05):
            cov = []
            for o in G.DEV_ORIGINS:
                E = np.stack(fc[v][o]); m, s = E.mean(0), E.std(0) + 1e-9
                cov.append(np.mean(np.abs(G.fold(mag, o)[1] - m) <= 1.96 * l * s))
            gap = abs(np.mean(cov) - 0.95)
            if gap < best_gap: best_l, best_gap = float(l), gap
        lam[v] = best_l
    print("ensemble lambda (dev-calibrated):", lam, flush=True)

    # test-fold predictions for every method
    preds, per_seed, cover, gpr_t = {}, {}, {}, []
    for v in variants:
        preds[f"{v} (ensemble)"] = {o: np.stack(fc[v][o]).mean(0) for o in G.TEST_ORIGINS}
        per_seed[v] = {o: fc[v][o] for o in G.TEST_ORIGINS}
        cov = {lv: [] for lv in LEVELS}
        for o in G.TEST_ORIGINS:
            E = np.stack(fc[v][o]); m, s = E.mean(0), E.std(0) + 1e-9
            tr = G.fold(mag, o)[1]
            for lv in LEVELS:
                cov[lv].append(np.mean(np.abs(tr - m) <= norm.ppf(0.5 + lv / 2) * lam[v] * s))
        cover[v] = {str(lv): float(100 * np.mean(c)) for lv, c in cov.items()}
    for o in G.TEST_ORIGINS:
        hist, _ = G.fold(mag, o)
        B, _ = G.baselines(hist, G.N_FC, K, p)
        for k, val in B.items():
            preds.setdefault(k, {})[o] = val
        pg, _, tg = pf_gpr(freqs, hist)
        preds.setdefault("PF 1D GPR", {})[o] = pg; gpr_t.append(tg)
        print(f"  baselines done for origin {o}", flush=True)

    # metrics
    def summ(method_preds, origins):
        M = [G.metrics(method_preds[o], G.fold(mag, o)[1]) for o in origins]
        S = np.array([m["step_rmse"] for m in M])
        return dict(rmse_mean=float(np.mean([m["rmse"] for m in M])),
                    rmse_sd=float(np.std([m["rmse"] for m in M])),
                    r2dem_mean=float(np.mean([m["r2_dem"] for m in M])),
                    r2dem_sd=float(np.std([m["r2_dem"] for m in M])),
                    step_mean=S.mean(0).tolist(), per_fold_rmse=[m["rmse"] for m in M]), S
    rng = np.random.default_rng(0)
    nT = len(G.TEST_ORIGINS)
    boots = rng.integers(0, nT, size=(NBOOT, nT))
    summary, steps = {}, {}
    for k, mp_ in preds.items():
        summary[k], steps[k] = summ(mp_, G.TEST_ORIGINS)
        b = steps[k][boots].mean(1)                     # (NBOOT, 20)
        summary[k]["step_ci_lo"] = np.percentile(b, 2.5, 0).tolist()
        summary[k]["step_ci_hi"] = np.percentile(b, 97.5, 0).tolist()
        summary[k]["excl_submitted_rmse_mean"] = summ(
            mp_, [o for o in G.TEST_ORIGINS if o != G.SUBMITTED_ORIGIN])[0]["rmse_mean"]
        summary[k]["submitted_fold"] = G.metrics(mp_[G.SUBMITTED_ORIGIN],
                                                 G.fold(mag, G.SUBMITTED_ORIGIN)[1])
    # seed variability of single models
    for v in variants:
        seed_rmse = [[G.metrics(per_seed[v][o][s], G.fold(mag, o)[1])["rmse"]
                      for o in G.TEST_ORIGINS] for s in range(len(SEEDS))]
        sr = np.array(seed_rmse).mean(1)
        summary[f"{v} (single seed)"] = dict(rmse_mean=float(sr.mean()), rmse_sd_over_seeds=float(sr.std()))
    simple = [k for k in sel["baseline_dev_rmse"] if k != "Climatology"]
    best_simple = min(simple, key=lambda k: sel["baseline_dev_rmse"][k])
    look = {}
    for v in variants:
        key = f"{v} (ensemble)"
        look[v] = dict(
            h_skill=lookahead((steps[key] - steps["Climatology"])[boots].mean(1)),
            h_adv=lookahead((steps[key] - steps[best_simple])[boots].mean(1)),
            best_simple_baseline=best_simple)
    out = dict(selection=sel, lambda_dev=lam, coverage_test=cover, lookahead=look,
               summary=summary,
               timing=dict(gru={v: dict(train_s_mean=float(np.mean([t[0] for t in times[v]])),
                                       infer_ms_mean=float(1e3 * np.mean([t[1] for t in times[v]])))
                                for v in variants},
                           pf_gpr_s_mean=float(np.mean(gpr_t))),
               test_origins=G.TEST_ORIGINS, seeds=SEEDS,
               run_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(out, open(G.OUT / "test_summary.json", "w"), indent=1, default=str)
    np.savez_compressed(G.OUT / "test_forecasts.npz",
                        **{k.replace(" ", "_"): np.stack([v[o] for o in G.TEST_ORIGINS])
                           for k, v in preds.items()},
                        **{f"seeds__{v.replace(' ', '_')}":
                           np.stack([np.stack(per_seed[v][o]) for o in G.TEST_ORIGINS])
                           for v in variants},
                        hist=np.stack([G.fold(mag, o)[0] for o in G.TEST_ORIGINS]),
                        truth=np.stack([G.fold(mag, o)[1] for o in G.TEST_ORIGINS]),
                        origins=np.array(G.TEST_ORIGINS))
    print("\n=== TEST (9 folds) ===")
    for k in sorted(summary, key=lambda k: summary[k]["rmse_mean"]):
        s = summary[k]
        if "step_mean" in s:
            print(f"{k:28s} RMSE {s['rmse_mean']:.3f}±{s['rmse_sd']:.3f}  R2dem {s['r2dem_mean']:+.3f}  "
                  f"h1 {s['step_mean'][0]:.3f}  h5 {s['step_mean'][4]:.3f}  h20 {s['step_mean'][19]:.3f}  "
                  f"excl160 {s['excl_submitted_rmse_mean']:.3f}  fold160 {s['submitted_fold']['rmse']:.3f}")
        else:
            print(f"{k:28s} RMSE {s['rmse_mean']:.3f} (SD over seeds {s['rmse_sd_over_seeds']:.3f})")
    print("lookahead:", look); print("coverage:", cover); print("timing:", out["timing"])


if __name__ == "__main__":
    main()
