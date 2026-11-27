"""[revision 2026-11-26] Recompute Table I (Multisine vs VNA on the static
equivalent circuit) from the raw data plotted in Fig. 4.

Inputs (submission-revision version/):
  不同频点结果-电路.xlsx       columns: N=20 (21 tones), N=40, N=70, N=100, VNA  [dB]
  F-A-P-100k-10M-LOG-{21,40,70,100}.xlsx   column 1: tone frequencies [Hz]
  F_VNA.xlsx                   VNA frequencies [Hz]
Each Multisine configuration is compared at its own tones with the VNA
response interpolated linearly in log10(f).  Error e = Multisine - VNA.
Outputs: table1_vna_metrics.json (full band, three sub-bands, bias and SD).
"""
import json
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).resolve().parents[1] / "data" / "circuit_vna"
res = pd.read_excel(D / "不同频点结果-电路.xlsx", header=None)
f_vna = pd.read_excel(D / "F_VNA.xlsx", header=None)[0].to_numpy(float)
vna = res[4].to_numpy(float)
grids = {20: "F-A-P-100k-10M-LOG-21.xlsx", 40: "F-A-P-100k-10M-LOG-40.xlsx",
         70: "F-A-P-100k-10M-LOG-70.xlsx", 100: "F-A-P-100k-10M-LOG-100.xlsx"}
BANDS = {"0.1-0.2 MHz": (1e5, 2e5), "0.2-2 MHz": (2e5, 2e6), "2-10 MHz": (2e6, 1.0001e7)}


def metrics(e):
    return dict(n=int(e.size), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                max=float(np.max(np.abs(e))), bias=float(np.mean(e)), sd=float(np.std(e)))


out = {}
for col, (n, g) in enumerate(grids.items()):
    f = pd.read_excel(D / g, header=None)[1].to_numpy(float)
    y = res[col].dropna().to_numpy(float)
    assert f.size == y.size, (n, f.size, y.size)
    e = y - np.interp(np.log10(f), np.log10(f_vna), vna)
    out[n] = {"tones": int(f.size), "full": metrics(e)}
    for b, (lo, hi) in BANDS.items():
        out[n][b] = metrics(e[(f >= lo) & (f < hi)])

pathlib.Path(__file__).with_name("table1_vna_metrics.json").write_text(json.dumps(out, indent=1))
for n, r in out.items():
    fm = r["full"]
    print(n, r["tones"], "RMSE %.3f MAE %.3f Max %.3f bias %+.3f SD %.3f" % (fm["rmse"], fm["mae"], fm["max"], fm["bias"], fm["sd"]),
          " | " + "  ".join("%s %.3f/%.3f" % (b, r[b]["rmse"], r[b]["max"]) for b in BANDS))
