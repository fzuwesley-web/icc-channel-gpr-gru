"""export_results.py — export the round-4 GPR results to the supplementary
material (metrics CSV, fitted hyperparameters JSON) and print the values used
in Tables II, III and V.   [revision, round 4]"""
import importlib.util, sys, pickle, json, pathlib, re
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("gpr_src", HERE / "gpr_models.py")
g = importlib.util.module_from_spec(sp); sys.modules["gpr_src"] = g; sp.loader.exec_module(g)
mm = sys.modules["__main__"]
for n in dir(g):
    if not n.startswith("__") and not hasattr(mm, n):
        setattr(mm, n, getattr(g, n))
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "export"
OUT.mkdir(parents=True, exist_ok=True)
rows, hyp = [], {}
for ds in ("exvivo", "invivo"):
    R = pickle.load(open(HERE / f"{ds}_result.pkl", "rb"))
    prot = {"In-sample": {r["label"]: r for r in R["compare_results"]}}
    for k, v in R["holdout_results"].items():
        if k.endswith("(holdout)"): prot.setdefault("Checkerboard", {})[k.split(" (")[0]] = v
    for k, v in R["block_results"].items():
        if k == "_meta" or v is None: continue
        p = "Time block" if "time-block" in k else "Frequency block"
        prot.setdefault(p, {})[k.split(" (")[0]] = v
    for p, d in prot.items():
        for scheme, m in d.items():
            row = dict(dataset=ds, protocol=p, scheme=scheme, rmse_dB=m["rmse"], mae_dB=m["mae"],
                       r2=m["r2"], r2_demeaned=m["r2_demeaned"])
            for lv, c in sorted(m["calibration"].items(), key=lambda t: float(t[0])):
                row[f"cov{int(round(100 * float(lv)))}"] = c
            rows.append(row)
    gp = R["models"]["2D"].gpr_mag; k = gp.kernel_
    names = [h.name for h in k.hyperparameters if not h.fixed]
    s = str(k)
    hyp[ds] = dict(kernel=s, lml=float(gp.log_marginal_likelihood_value_),
                   theta=dict(zip(names, np.exp(k.theta).tolist())),
                   ell_f=float(re.search(r"RBF\(length_scale=([0-9.e-]+)", s).group(1)),
                   C_p=float(re.search(r"([0-9.e-]+)\*\*2 \* ExpSineSquared", s).group(1)) ** 2,
                   ell_p=float(re.search(r"ExpSineSquared\(length_scale=([0-9.e-]+)", s).group(1)),
                   T=float(re.search(r"periodicity=([0-9.e-]+)", s).group(1)),
                   C_m=float(re.findall(r"([0-9.e-]+)\*\*2 \* RBF", s)[-1]) ** 2,      # [round 5] non-periodic RBF term
                   ell_m=float(re.findall(r"RBF\(length_scale=([0-9.e-]+)\)\) \+ White", s)[-1]),
                   noise_sigma_dB=float(R["models"]["2D"].noise_sigma_dB),
                   at_bound=int((np.isclose(k.theta, k.bounds[:, 0], atol=1e-5) |
                                 np.isclose(k.theta, k.bounds[:, 1], atol=1e-5)).sum()),
                   mean_order=int(R["mean_fn"].order), mean_r2=float(R["mean_fn"].r2),
                   coarse_kernel=str(getattr(R["models"]["2D"], "coarse_kernel_", "")),
                   optimizer="coarse-to-fine deterministic multistart with staged full-resolution refinement (round 5b)",
                   t_2d_s=R["profiling"]["t_2d"], t_pf_s=R["profiling"]["t_pf"], t_pt_s=R["profiling"]["t_pt"],
                   mem_peak_MB=R["profiling"]["mem_peak"] / 2**20 if R["profiling"]["mem_peak"] > 1e6 else R["profiling"]["mem_peak"])
    # exact hyperparameters from theta (more digits than the repr)
    th = dict(zip(names, np.exp(k.theta)))
    for key, pat in (("ell_f", "k1__k1__kernel__length_scale"), ("T", "periodicity")):
        cand = [v for n_, v in th.items() if n_.endswith(pat)]
        if cand: hyp[ds][key] = float(cand[0])
df = pd.DataFrame(rows)
df.to_csv(OUT / "gpr_metrics_all_protocols.csv", index=False, float_format="%.6g")
json.dump(hyp, open(OUT / "fitted_hyperparameters.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
print(json.dumps({d: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in h.items()
                      if k not in ("kernel", "theta", "coarse_kernel")} for d, h in hyp.items()}, indent=1))
