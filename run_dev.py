"""run_dev.py — hyperparameter and variant selection on DEVELOPMENT folds only.

Selection rule (fixed before any test-fold run):
  * Baselines: harmonic K in {1,2,3,4} and AR order p in {2,4,6,8,10,12,15,20}
    minimizing mean forecast RMSE over the development folds.
  * GRU: for each variant (plain GRU; harmonic(K*)+GRU residual), the
    (H, W, training regime) minimizing mean forecast RMSE over
    development folds x 3 seeds.
  * Main method: the variant with the lower selected mean RMSE.
Output: cv_results/dev_raw.json, cv_results/dev_selection.json (locked).
"""
import itertools
import json
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import gru_cv as G  # noqa: E402

H_GRID, W_GRID, SEEDS = (16, 32, 64), (10, 13, 20), (0, 1, 2)
K_GRID, P_GRID = (1, 2, 3, 4), (2, 4, 6, 8, 10, 12, 15, 20)


def _task(args):
    import torch
    torch.set_num_threads(1)
    variant, H, W, o, seed = args
    _, mag = G.load_invivo()
    hist, true = G.fold(mag, o)
    fcs, diag = G.gru_forecasts(hist, W, H, seed, residual_K=variant)
    return [dict(variant=("GRU" if variant is None else f"Harm{variant}+GRU"),
                 H=H, W=W, origin=o, seed=seed, regime=k,
                 **G.metrics(v, true),
                 train_s=diag[k].get("train_s"), infer_s=diag[k].get("infer_s"))
            for k, v in fcs.items()]


def main():
    G.OUT.mkdir(exist_ok=True)
    _, mag = G.load_invivo()
    # 1) baseline hyperparameters
    folds = [G.fold(mag, o) for o in G.DEV_ORIGINS]
    k_score = {K: np.mean([G.metrics(G.harmonic_fit(h, K, G.fundamental(h), G.N_FC)[1], t)["rmse"]
                           for h, t in folds]) for K in K_GRID}
    p_score = {p: np.mean([G.metrics(G.ar_forecast(h, p, G.N_FC), t)["rmse"] for h, t in folds])
               for p in P_GRID}
    K_star = min(k_score, key=k_score.get); p_star = min(p_score, key=p_score.get)
    print("harmonic K dev RMSE:", {k: round(v, 3) for k, v in k_score.items()}, "-> K*", K_star)
    print("AR p dev RMSE:", {k: round(v, 3) for k, v in p_score.items()}, "-> p*", p_star)
    # 2) GRU grid
    tasks = list(itertools.product((None, K_star), H_GRID, W_GRID, G.DEV_ORIGINS, SEEDS))
    t0 = time.time()
    with mp.get_context("spawn").Pool(6) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks), 1):
            rows += r
            if i % 18 == 0:
                print(f"  {i}/{len(tasks)} tasks  {time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(G.OUT / "dev_raw.json", "w"))
    # 3) selection
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["H"], r["W"], r["regime"])].append(r["rmse"])
    best = {}
    for (v, H, W, reg), xs in agg.items():
        m = float(np.mean(xs))
        if v not in best or m < best[v]["dev_rmse"]:
            best[v] = dict(H=H, W=W, regime=reg, dev_rmse=m, dev_sd=float(np.std(xs)), n=len(xs))
    main_v = min(best, key=lambda v: best[v]["dev_rmse"])
    # baseline dev scores at selected hyperparameters, for the record
    base_dev = {}
    for h, t in folds:
        B, _ = G.baselines(h, G.N_FC, K_star, p_star)
        for k, v in B.items():
            base_dev.setdefault(k, []).append(G.metrics(v, t)["rmse"])
    sel = dict(rule=__doc__.strip().splitlines()[2:9], K_star=K_star, p_star=p_star,
               k_scores=k_score, p_scores=p_score, gru=best, main_variant=main_v,
               baseline_dev_rmse={k: float(np.mean(v)) for k, v in base_dev.items()},
               dev_origins=G.DEV_ORIGINS, seeds=SEEDS, wall_s=time.time() - t0,
               locked_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(sel, open(G.OUT / "dev_selection.json", "w"), indent=1, default=str)
    print(json.dumps({k: sel[k] for k in ("K_star", "p_star", "gru", "main_variant", "baseline_dev_rmse")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
