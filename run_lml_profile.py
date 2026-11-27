"""run_lml_profile.py — profile log marginal likelihood of the 2D GPR over the
cardiac period T (supplementary Fig. S3).   [revision, round 3]

For each T on a log-spaced grid over the admissible range [0.4, 2.0] s, T is
held fixed and the remaining five free hyperparameters are re-optimised
(L-BFGS-B, 200 iterations), warm-started from the neighbouring grid point.
Three constrained fits are run at every grid point (warm starts from the
neighbouring point in increasing and in decreasing T, and a start from the
unconstrained optimum) and the largest LML is kept, which limits path
dependence.  This is a true profile,
unlike the conditional slice (other parameters fixed) shown in the first
revision draft.

Resolution: every 4th frequency (25 of 100) x all 80 instants = 2000 points,
so that 2 x 33 constrained fits per dataset are affordable.  The noise level is
fixed to the full-resolution D4 estimate, as in the main fits.  The full
multistart fit at the same resolution marks the unconstrained optimum.
Output: lml_profiles_round3.npz
"""
import importlib.util as ilu, sys, time, warnings, pathlib, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, WhiteKernel, ConstantKernel as C

HERE = pathlib.Path(__file__).resolve().parent
sp = ilu.spec_from_file_location("gpr_src", HERE / "gpr_models.py")
G = ilu.module_from_spec(sp); sys.modules["gpr_src"] = G; sp.loader.exec_module(G)
SK = G.SlicedKernel
DATA = next(p for p in (HERE / "data", HERE.parent / "data") if (p / "f_LOG100.xlsx").exists())
freqs = pd.read_excel(DATA / "f_LOG100.xlsx", header=None).values.flatten()
SETS = {
    "exvivo": dict(file="invitro_Rx_80T_100f_3.xlsx", sl=slice(0, 80), tx=0.330507, ref_T=1.0, ref="pump 60.0 bpm"),
    "invivo": dict(file="invivo_Rx_200T_100f_3.xlsx", sl=slice(100, 180), tx=0.052738597, ref_T=60 / 95.0, ref="channel FFT / T* 95 bpm"),
}
TG = np.exp(np.linspace(np.log(0.4), np.log(2.0), 33))
FI = np.arange(0, 100, 4)


def kernel(T, nv, fixed_T=True):
    ess = ExpSineSquared(0.15, T, (0.05, 20.0), "fixed" if fixed_T else (0.4, 2.0))
    return (SK(RBF(0.5, (0.02, 5.0)), 0) * (SK(C(1.0, (1e-2, 1e2)) * ess, 1)
            + SK(C(0.5, (1e-3, 1e2)) * RBF(2.0, (0.05, 20.0)), 1)) + WhiteKernel(nv, "fixed"))  # [round 5] RBF


def main():
    out = {}
    for name, cfg in SETS.items():
        Ra = pd.read_excel(DATA / cfg["file"], header=None).values.T[:, cfg["sl"]]
        mag = 20 * np.log10(np.maximum(Ra / cfg["tx"], 1e-15))
        times = np.arange(mag.shape[1]) * 0.05
        sig = G.estimate_measurement_noise(mag, axis=1)            # full-resolution D4
        mfn = G.FrequencyMeanFunction().fit(freqs, mag)
        R = mfn.residual(freqs, mag)[FI]
        y = R.ravel(); nv = float((sig / y.std()) ** 2)
        A, B = np.meshgrid(np.log10(freqs[FI]), times, indexing="ij")
        X = np.column_stack([A.ravel(), B.ravel()])
        t0 = time.time()
        full = GPR(kernel(cfg["ref_T"], nv, fixed_T=False), optimizer=G._multistart_lbfgs_optimizer,
                   n_restarts_optimizer=0, normalize_y=True, alpha=1e-6).fit(X, y)
        import re
        T_star = float(re.search(r"periodicity=([0-9.]+)", str(full.kernel_)).group(1))
        lml_star = float(full.log_marginal_likelihood_value_)
        print(f"[{name}] multistart: T*={T_star:.4f} s LML={lml_star:.1f} ({time.time()-t0:.0f}s)", flush=True)
        # theta of the unconstrained optimum without the period entry
        # [round 4] the ex vivo peak in T is narrow (near-deterministic channel,
        # sigma_n = 0.005 dB), so the log grid alone can step over it: add points
        # densely around T* and its multiples within the admissible range.
        dens = [T_star * m * (1 + d) for m in (0.5, 1.0, 2.0, 3.0)
                for d in (-0.02, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, 0.02)]
        TGd = np.unique(np.round(np.concatenate([TG, [x for x in dens if 0.4 <= x <= 2.0]]), 6))
        names = [h.name for h in full.kernel_.hyperparameters if not h.fixed]
        jT = [i for i, n_ in enumerate(names) if n_.endswith("periodicity")][0]
        th_star = np.delete(full.kernel_.theta, jT)
        sweeps = []
        for mode in ("up", "down", "from_optimum"):
            order = range(len(TGd) - 1, -1, -1) if mode == "down" else range(len(TGd))
            lml = np.full(len(TGd), np.nan); th = th_star if mode == "from_optimum" else None
            for j in order:
                k = kernel(TGd[j], nv)
                if th is not None:
                    k = k.clone_with_theta(th)
                m = GPR(k, optimizer=G._capped_lbfgs_optimizer, n_restarts_optimizer=0,
                        normalize_y=True, alpha=1e-6).fit(X, y)
                lml[j] = m.log_marginal_likelihood_value_
                th = th_star if mode == "from_optimum" else m.kernel_.theta
            sweeps.append(lml)
            print(f"[{name}] sweep {mode} done ({time.time()-t0:.0f}s)", flush=True)
        prof = np.nanmax(np.vstack(sweeps), 0)
        loc = [j for j in range(1, len(TGd) - 1) if prof[j] > prof[j - 1] and prof[j] > prof[j + 1]]
        out[name] = dict(T=TGd, profile=prof, sweep_up=sweeps[0], sweep_down=sweeps[1],
                         sweep_from_optimum=sweeps[2], T_star=T_star,
                         lml_star=lml_star, ref_T=cfg["ref_T"], local_max_T=TGd[loc], sigma_dB=sig,
                         n_points=len(y))
        print(f"[{name}] profile argmax T={TGd[np.argmax(prof)]:.4f} s, max={prof.max():.1f}; "
              f"local maxima at {np.round(TGd[loc], 3).tolist()}", flush=True)
    np.savez(HERE / "lml_profiles_round3.npz",
             **{f"{n}__{k}": np.asarray(v) for n, d in out.items() for k, v in d.items()})


if __name__ == "__main__":
    main()
