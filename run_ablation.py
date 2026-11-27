
"""run_ablation.py (v6) — kernel ablation at full time resolution (Table IV).
[v5, 2026-11-09] v4 subsampled time to 100 ms and estimated the noise on that grid,
which inflated sigma_n to 0.141 dB (fourth differences leak the cardiac oscillation
at 100 ms). v5 keeps 50 ms, uses every 4th frequency, the time-block split, and the
full-resolution noise estimate."""
import importlib.util as ilu, sys, json, time, warnings, pathlib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, WhiteKernel, ConstantKernel as C
from scipy.optimize import minimize as _min

RV = pathlib.Path(__file__).resolve().parent
sp = ilu.spec_from_file_location("gpr_src", RV/"gpr_models.py")
G = ilu.module_from_spec(sp); sys.modules["gpr_src"] = G; sp.loader.exec_module(G)
SK, MF = G.SlicedKernel, G.FrequencyMeanFunction

D = RV/"data" if (RV/"data").exists() else RV.parent/"data"
freqs = pd.read_excel(D/"f_LOG100.xlsx", header=None).values.flatten()
Ra = pd.read_excel(D/"invivo_Rx_200T_100f_3.xlsx", header=None).values.T[:, 100:180]
times = np.arange(0.0, 10.0, 0.05)[100:180]
mag = 20*np.log10(np.maximum(Ra/0.052738597, 1e-15))
freqs_s, times_s, mag_s = freqs[::4], times, mag[::4]      # [v5] 25 freq x all 80 instants (50 ms), as in the profile
HR = 1/1.5
N_TR = 60                                                  # [v5] train 5.00-7.95 s, test 8.00-8.95 s (= time-block protocol)

def opt200(f, t0, bounds):
    r = _min(f, t0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter":200})
    return r.x, r.fun

def kf():  return SK(RBF(0.5, (0.02, 5.0)), 0)
def ess(): return C(1.0,(1e-2,1e2))*ExpSineSquared(0.15, HR, (0.05,20.0), (0.4,2.0))
def mat(): return C(0.5,(1e-3,1e2))*Matern(2.0, (0.05,20.0), nu=1.5)
def rbf_t(): return C(0.5,(1e-3,1e2))*RBF(0.5, (0.05,20.0))

VARIANTS = {
 "A. product, ESS+RBF(t) (this work)":  lambda nv: kf()*(SK(ess(),1)+SK(rbf_t(),1)) + WhiteKernel(nv,"fixed"),
 "B. product, ESS only":                lambda nv: kf()*SK(ess(),1) + WhiteKernel(nv,"fixed"),
 "C. product, RBF(t) only":             lambda nv: kf()*SK(rbf_t(),1) + WhiteKernel(nv,"fixed"),
 "D. product, Matern only":             lambda nv: kf()*SK(mat(),1) + WhiteKernel(nv,"fixed"),
 "E. product, ESS+Matern (first revision)": lambda nv: kf()*(SK(ess(),1)+SK(mat(),1)) + WhiteKernel(nv,"fixed"),
 "F. ADDITIVE (not product)":           lambda nv: kf()+SK(ess(),1)+SK(rbf_t(),1) + WhiteKernel(nv,"fixed"),
 "G. no frequency kernel":              lambda nv: SK(ess(),1)+SK(rbf_t(),1) + WhiteKernel(nv,"fixed"),
}

def build_X(lf, tt):
    A,B = np.meshgrid(lf, tt, indexing="ij"); return np.column_stack([A.ravel(), B.ravel()])

out = {}
mfn = MF().fit(freqs_s, mag_s[:, :N_TR])
Rtr = mfn.residual(freqs_s, mag_s[:, :N_TR])
sig = G.estimate_measurement_noise(mag, axis=1)   # [v5] full-resolution D4 estimate (0.0173 dB), as in the main fits
nv  = float((sig/Rtr.std())**2)
Xtr = build_X(np.log10(freqs_s), times_s[:N_TR])
Xte = build_X(np.log10(freqs_s), times_s[N_TR:])
Yte = mag_s[:, N_TR:]

for name, kb in VARIANTS.items():
    t0 = time.perf_counter()
    m = GPR(kernel=kb(nv), n_restarts_optimizer=0, normalize_y=True, alpha=1e-6, optimizer=G._multistart_lbfgs_optimizer)  # [round 3]
    m.fit(Xtr, Rtr.ravel())
    ins = m.predict(Xtr).reshape(Rtr.shape)
    pr, sd = m.predict(Xte, return_std=True)
    pr = mfn.restore(freqs_s, pr.reshape(Yte.shape)); sd = sd.reshape(Yte.shape)
    met = G.compute_metrics(Yte, pr, sd)
    kk = m.kernel_
    atb = int((np.isclose(kk.theta, kk.bounds[:,0], atol=1e-5) |
               np.isclose(kk.theta, kk.bounds[:,1], atol=1e-5)).sum())
    out[name] = dict(lml=float(m.log_marginal_likelihood_value_), n_hyper=len(kk.theta),
                     at_bound=atb, rmse_insample=float(np.sqrt(np.mean((ins-Rtr)**2))),
                     rmse_block=met["rmse"], r2dem_block=met["r2_demeaned"],
                     cov95_block=met["cov95"], secs=time.perf_counter()-t0,
                     kernel=str(kk))
    print(f"[{name}] LML={out[name]['lml']:9.1f} p={len(kk.theta)} bound={atb} "
          f"block RMSE={met['rmse']:.4f} R2dem={met['r2_demeaned']:+.4f} ({out[name]['secs']:.0f}s)", flush=True)

json.dump(out, open(RV/"kernel_ablation.json","w"), indent=1)
print("ABLATION DONE")
