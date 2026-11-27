"""run_sensitivity.py — sensitivity analyses for the GRU protocol (GRU_REVISION_PLAN.md §3.8).

Uses the configuration LOCKED by run_dev.py (no re-selection).  All results are
reported regardless of outcome.

  S-a  non-overlapping test folds o = 100, 120, 140, 160, 180 (from test_forecasts.npz)
  S-b  4 s history (80 frames), 1 s forecast, o = 100 ... 180
  S-c  expanding window (all frames before o), 1 s forecast, o = 100 ... 180
  S-d  3 s history, 2 s forecast (40 steps), o = 100 ... 160

Output: cv_results/sensitivity_summary.json
"""
import json
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import gru_cv as G  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
SCEN = {
    "S-b 4 s history":      dict(hist=80,   n_fc=20, origins=tuple(range(100, 181, 10))),
    "S-c expanding window": dict(hist=None, n_fc=20, origins=tuple(range(100, 181, 10))),
    "S-d 2 s horizon":      dict(hist=60,   n_fc=40, origins=tuple(range(100, 161, 10))),
}
NONOVERLAP = (100, 120, 140, 160, 180)


def get_fold(mag, o, hist_len, n_fc):
    h = mag[:, :o] if hist_len is None else mag[:, o - hist_len:o]
    return h, mag[:, o:o + n_fc]


def _task(args):
    import torch
    torch.set_num_threads(1)
    scen, vname, K, H, W, regime, o, seed = args
    cfg = SCEN[scen]
    _, mag = G.load_invivo()
    hist, _ = get_fold(mag, o, cfg["hist"], cfg["n_fc"])
    G.N_HIST, G.N_FC = hist.shape[1], cfg["n_fc"]       # module globals read at call time
    if regime == "ES":
        fc, d = G.gru_forecast(hist, W, H, seed, residual_K=K)
        tr = d["train_s"]
    else:
        E = int(regime[1:])
        fcs, d = G.gru_forecasts(hist, W, H, seed, residual_K=K, ckpts=(E,), with_es=False)
        fc, tr = fcs[regime], d[regime]["train_s"]
    return scen, vname, o, seed, fc, tr


def summarize(preds, truths, origins):
    M = [G.metrics(preds[o], truths[o]) for o in origins]
    S = np.array([m["step_rmse"] for m in M])
    return dict(rmse_mean=float(np.mean([m["rmse"] for m in M])),
                rmse_sd=float(np.std([m["rmse"] for m in M])),
                r2dem_mean=float(np.mean([m["r2_dem"] for m in M])),
                step_mean=S.mean(0).tolist(), per_fold_rmse=[m["rmse"] for m in M],
                n_folds=len(origins))


def main():
    sel = json.load(open(G.OUT / "dev_selection.json"))
    K, p = sel["K_star"], sel["p_star"]
    _, mag = G.load_invivo()
    variants = {v: (None if v == "GRU" else K, c["H"], c["W"], c["regime"]) for v, c in sel["gru"].items()}
    out = dict(locked_selection=sel["gru"], K_star=K, p_star=p, scenarios={})

    # S-a: re-score the main test forecasts on non-overlapping folds only
    Z = np.load(G.OUT / "test_forecasts.npz")
    orig = list(Z["origins"]); idx = [orig.index(o) for o in NONOVERLAP]
    truths = {o: Z["truth"][i] for o, i in zip(NONOVERLAP, idx)}
    sa = {}
    for k in Z.files:
        if k in ("hist", "truth", "origins") or k.startswith("seeds__"):
            continue
        sa[k.replace("_", " ")] = summarize({o: Z[k][i] for o, i in zip(NONOVERLAP, idx)}, truths, NONOVERLAP)
    out["scenarios"]["S-a non-overlapping folds"] = dict(origins=NONOVERLAP, methods=sa)

    # S-b, S-c, S-d: new fits with the locked configuration
    tasks = [(sc, v, *variants[v], o, s) for sc, cfg in SCEN.items()
             for v in variants for o in cfg["origins"] for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(6) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks), 1):
            res.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)} fits  {time.time() - t0:.0f}s", flush=True)
    for sc, cfg in SCEN.items():
        tr = {o: get_fold(mag, o, cfg["hist"], cfg["n_fc"])[1] for o in cfg["origins"]}
        preds = {}
        for v in variants:
            ens = {o: np.mean([r[4] for r in res if r[0] == sc and r[1] == v and r[2] == o], 0)
                   for o in cfg["origins"]}
            preds[f"{v} (ensemble)"] = ens
        for o in cfg["origins"]:
            hist, _ = get_fold(mag, o, cfg["hist"], cfg["n_fc"])
            B, _ = G.baselines(hist, cfg["n_fc"], K, p)
            for k, val in B.items():
                preds.setdefault(k, {})[o] = val
        out["scenarios"][sc] = dict(
            origins=cfg["origins"], hist_frames=cfg["hist"] or "expanding", n_fc=cfg["n_fc"],
            methods={k: summarize(v, tr, cfg["origins"]) for k, v in preds.items()},
            gru_train_s_mean=float(np.mean([r[5] for r in res if r[0] == sc])))
    out["wall_s"] = time.time() - t0
    out["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(out, open(G.OUT / "sensitivity_summary.json", "w"), indent=1, default=str)
    for sc, d in out["scenarios"].items():
        print(f"\n=== {sc} ({len(d['origins'])} folds) ===")
        for k in sorted(d["methods"], key=lambda k: d["methods"][k]["rmse_mean"]):
            m = d["methods"][k]
            print(f"  {k:26s} RMSE {m['rmse_mean']:.3f}±{m['rmse_sd']:.3f}  R2dem {m['r2dem_mean']:+.3f}  "
                  f"h1 {m['step_mean'][0]:.3f}  hlast {m['step_mean'][-1]:.3f}")


if __name__ == "__main__":
    main()
