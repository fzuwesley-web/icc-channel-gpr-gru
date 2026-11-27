"""Clean wall-clock timing of the in vivo 2D main fit (round 5b), run alone."""
import importlib.util as ilu, sys, time, json, warnings, numpy as np, pandas as pd, pathlib
warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).parent
DD = D/"data" if (D/"data").exists() else D.parent/"data"
sp = ilu.spec_from_file_location("gpr_src", D/"gpr_models.py"); G = ilu.module_from_spec(sp); sys.modules["gpr_src"]=G; sp.loader.exec_module(G)
freqs = pd.read_excel(DD/"f_LOG100.xlsx", header=None).values.flatten()
Ra = pd.read_excel(DD/"invivo_Rx_200T_100f_3.xlsx", header=None).values.T[:, 100:180]
mag = 20*np.log10(np.maximum(Ra/0.052738597, 1e-15)); t = 5.0 + np.arange(80)*0.05
mfn = G.FrequencyMeanFunction().fit(freqs, mag); R = mfn.residual(freqs, mag)
t0 = time.perf_counter(); m = G.JointGPR2D(freqs, t, heart_rate_hz=1.5, n_restarts=5); m.fit(R); dt = time.perf_counter()-t0
out = dict(lml=float(m.gpr_mag.log_marginal_likelihood_value_), kernel=str(m.gpr_mag.kernel_), t_2d_s=dt)
print(out); json.dump(out, open(D/"export_round5b/invivo_2d_timing_clean.json","w"), indent=1)
