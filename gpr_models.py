"""
=============================================================================
面向无线心脏起搏器的房室间时变信道 GPR 建模  v3  (amplitude only)
Dynamic Intracardiac Channel Magnitude Estimation via 2D Joint GPR

[v2 → v3 架构升级]
  v2: 逐频点 1D GPR — N_freq 个独立模型，每个只使用时间轴
  v3: 2D 联合 GPR  — 幅度用一个模型，联合建模 (log10(f), t) 空间

2D 核函数结构:
  k((logf,t), (logf',t')) =
      [ C_f · RBF(logf, logf') ]
    × [ C_p · ExpSineSquared(t,t') + C_m · Matérn(t,t') ]
    + WhiteKernel

  频率轴 (logf): RBF 捕捉频谱的平滑变化
  时间轴 (t)  : ExpSineSquared 捕捉心动周期性
               Matérn 捕捉呼吸/体位等慢变趋势
  乘积耦合    : 频率与时间的不确定性被联合建模

输入（替换说明见文末）:
  Ra(f,t): 接收端时频幅度矩阵 (N_freq × N_time)，线性幅度
  Ta(f)  : 发射端频域幅度向量 (N_freq,)，线性幅度，时不变

信道计算:
  |H(f,t)|_dB = 20·log10( Ra(f,t) / Ta(f) )

输出:
  • 幅度信道 GPR 模型及核函数表达式
  • 可视化：热力图 / 时间切片 / 残差分布 / 各频点 RMSE
=============================================================================
"""

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel, RBF, Matern, ExpSineSquared, WhiteKernel, ConstantKernel as C
)
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize as _scipy_minimize
from tqdm import tqdm
import time
import tracemalloc
import warnings
warnings.filterwarnings('ignore')


# ── 带 maxiter 上限的 L-BFGS-B 优化器包装 ──────────────────────────────
# 针对弱信号（活体）数据，默认 L-BFGS-B 可能陷入超长迭代。限制每次
# restart 最多 N_MAX_OPT_ITER 次迭代，保证收敛速度可控。
N_MAX_OPT_ITER = 200   # [revision] 50 -> 200 (R3-7: 50 次迭代可能过早截断)

def _capped_lbfgs_optimizer(obj_func, initial_theta, bounds):
    """L-BFGS-B 优化器，带 maxiter 上限，防止单次 restart 无限迭代。"""
    res = _scipy_minimize(
        obj_func, initial_theta, method="L-BFGS-B", jac=True,
        bounds=bounds, options={'maxiter': N_MAX_OPT_ITER}
    )
    return res.x, res.fun


# ── [revision, round 3/4] 确定性多起点优化器 ───────────────────────────
# 第二轮（D4 噪声）活体 2D 拟合中，5 次无种子随机重启全部落在 T = 2×RR 的
# 局部极值（LML 14989.8），而从 T≈RR 出发可达 15267.1。第三轮仅在 T 上
# 布置起点（ℓ_p 仍取默认 0.15），活体正确，但离体落到 T = 2.0 s 边界
# （LML 29261.7，低于第二轮的 30154.6）。因此第四轮改为「粗到细」：
#   1) 粗搜索：在频率 1/4 抽样的网格上，起点 = T0 × {1/2, 3/4, 1, 3/2, 2, 3}
#      × ℓ_p ∈ {0.5, 2, 8}，再加 N_RANDOM_STARTS 个 RandomState(0) 随机起点；
#   2) 细化：全分辨率上从粗搜索最优解和默认初值各做一次 L-BFGS-B，取 LML 最大者。
# 全程确定性、可复现。
N_RANDOM_STARTS = 10
PERIOD_START_MULTIPLES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ELLP_STARTS = (0.5, 2.0, 8.0)
PERIOD_BOUNDS = (0.4, 2.0)
COARSE_FREQ_STEP = 4


def _period_index(bounds):
    b = np.asarray(bounds, float)
    lo, hi = np.log(PERIOD_BOUNDS[0]), np.log(PERIOD_BOUNDS[1])
    j = [i for i in range(len(b)) if np.isclose(b[i, 0], lo) and np.isclose(b[i, 1], hi)]
    return j[0] if j else None


def _grid_multistart_lbfgs_optimizer(obj_func, initial_theta, bounds):
    """Dense deterministic start grid (T multiples x ℓ_p values) + seeded random starts.
    In the ExpSineSquared parameterisation ℓ_p immediately precedes T in theta."""
    b = np.asarray(bounds, float)
    th0 = np.asarray(initial_theta, float)
    starts = [th0]
    jT = _period_index(bounds)
    if jT is not None:
        jl = jT - 1
        for mlt in PERIOD_START_MULTIPLES:
            for lp in ELLP_STARTS:
                th = th0.copy()
                th[jT] = np.clip(th0[jT] + np.log(mlt), b[jT, 0] + 1e-6, b[jT, 1] - 1e-6)
                th[jl] = np.clip(np.log(lp), b[jl, 0] + 1e-6, b[jl, 1] - 1e-6)
                if all(np.max(np.abs(th - s_)) > 1e-3 for s_ in starts):
                    starts.append(th)
    rng = np.random.RandomState(0)
    for _ in range(N_RANDOM_STARTS):
        starts.append(rng.uniform(b[:, 0], b[:, 1]))
    best_x, best_f = None, np.inf
    for th in starts:
        x, f = _capped_lbfgs_optimizer(obj_func, th, bounds)
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f


class RefineOptimizer:
    """Full-resolution refinement from the default initial value and given starts.
    A module-level callable class (not a closure) so that fitted models pickle."""
    def __init__(self, extra_starts):
        self.extra_starts = [np.asarray(e, float) for e in extra_starts]

    def __call__(self, obj_func, initial_theta, bounds):
        # [round 5b, 2026-11-09] staged refinement. At full resolution a free
        # L-BFGS-B run started from the coarse optimum can move T into a
        # lower local maximum before the amplitudes / length scales have
        # adapted to the denser data (observed: ex vivo time-block training
        # set, T 1.00 -> 0.60 s). For every start we therefore also run
        # (i) L-BFGS-B with T clamped at its start value, then (ii) free
        # L-BFGS-B from that point, and keep the candidate with the highest
        # LML. Selection remains purely by training log marginal likelihood.
        b = np.asarray(bounds, float)
        jT = _period_index(bounds)
        self.trace_ = []
        best_x, best_f = None, np.inf
        for th in [np.asarray(initial_theta, float)] + self.extra_starts:
            cands = [("direct", *_capped_lbfgs_optimizer(obj_func, th, bounds))]
            if jT is not None:
                bc = b.copy()
                bc[jT] = np.clip(th[jT], b[jT, 0], b[jT, 1])
                x1, _ = _capped_lbfgs_optimizer(obj_func, np.clip(th, bc[:, 0], bc[:, 1]), bc)
                cands.append(("staged", *_capped_lbfgs_optimizer(obj_func, x1, bounds)))
            for tag, x, f in cands:
                self.trace_.append((tag, np.exp(th).tolist(), np.exp(x).tolist(), -float(f)))
                if f < best_f:
                    best_x, best_f = x, f
        return best_x, best_f


def _make_refine_optimizer(extra_starts):
    return RefineOptimizer(extra_starts)


# backward-compatible name used by earlier scripts
_multistart_lbfgs_optimizer = _grid_multistart_lbfgs_optimizer

# ── 95% 置信区间精确系数 ────────────────────────────────────────────────
Z95 = norm.ppf(0.975)   # ≈ 1.96 （替代之前的近似值 2）


# ============================================================
# ① 仿真数据生成（实验时用真实数据替换，说明见文末）
# ============================================================


# ============================================================
# ② 信道矩阵计算
# ============================================================

def compute_channel(Ra, Ta):
    """
    由接收端/发射端幅度计算幅度信道特性。

    参数:
        Ra : (N_freq, N_time) 接收端线性幅度
        Ta : (N_freq,)        发射端线性幅度（时不变）

    返回:
        mag_dB : (N_freq, N_time) 幅度信道特性 [dB]，即 20·log10(Ra/Ta)
    """
    Ta_col = Ta[:, np.newaxis]
    ratio  = Ra / np.where(Ta_col > 1e-15, Ta_col, 1e-15)
    mag_dB = 20.0 * np.log10(np.where(ratio > 1e-15, ratio, 1e-15))

    print(f"[Channel] Magnitude range: {mag_dB.min():.2f} ~ {mag_dB.max():.2f} dB")
    return mag_dB


# ============================================================
# ②-B 显式均值函数 m(log10 f)   [revision / Fix B]
# ============================================================

class FrequencyMeanFunction:
    """
    GP 先验均值 m(log10 f)：对逐频点的时间均值拟合 log10(f) 的多项式。

    动机
    ----
    活体窗口内 98.9% 的总方差来自跨频率的静态频响（逐频均值跨度 29.2 dB），
    仅 1.1% 来自时变调制。原模型无显式均值函数，`normalize_y=True` 只减去
    一个全局标量，于是 GP 不得不用协方差去表示这段静态频响——表现为
    Matérn 项的 C_m 顶上界、ℓ_m = 12.4 s（比 4 s 窗口还长 3 倍），即该项
    在「冒充均值函数」。引入 m(·) 后 GP 只建模残差 ΔH。

    为什么必须是光滑参数化而非逐频经验均值
    ----------------------------------------
    频段块留出要求在**未观测频点**上评估，而逐频经验均值在留出频点上
    不存在。多项式可外推，经验均值不行。

    阶数选择与 max_order=3 的理由
    ------------------------------
    1..max_order 中按留一交叉验证（LOO）均方误差最小者。OLS 的 LOO 残差
    有闭式解 r_i/(1-h_ii)，无需真正重复拟合。

    max_order 固定为 3，不是由 LOO 自由选出来的。实测（活体，100 频点）
    各阶在三种频段留出位置上的外推/插值 RMSE：

        order   LOO RMSE   内插 70:90   内插 10:30   边缘外推 80:100
          3       0.688       0.526        0.621         0.951
          4       0.703       0.489        0.634         6.425
          6       0.550       1.117        1.013        36.561
          8       0.418       0.505        0.735        60.818
         20       0.440       3.801       14.178     4.78e+06

    LOO 最优在 8 阶附近，但 4 阶及以上在**频率轴边缘**出现多项式外推
    发散（Runge 现象）。频段块留出要求在未观测频段上求值，故必须选用
    在边缘仍然稳定的低阶。3 阶已移除约 99.5% 的跨频方差（逐频均值标准差
    9.08 -> 0.63 dB），足以解除 Matérn 项「冒充均值函数」的问题。

    职责划分
    --------
    m(·) 只负责**总体频响趋势**；残余的精细频率结构由核的 k_f = RBF(log f)
    建模。这是有意的分工，不是 m(·) 拟合不足。

    已知限制
    --------
    即便 3 阶，边缘频段（80:100）的外推 RMSE 仍有 0.95 dB，与残差场本身
    的量级相当。**频段块留出应使用内部频段**，边缘频段的结果需单独说明。
    """

    def __init__(self, max_order: int = 3):
        self.max_order = int(max_order)
        self.order = None
        self.coef = None
        self.loo_mse = None
        self.r2 = None
        self._lo = 0.0
        self._sc = 1.0

    def _design(self, lf):
        x = (np.asarray(lf, float) - self._lo) / self._sc
        return np.vander(x, self.order + 1, increasing=True)

    def fit(self, freqs, mag_dB):
        lf = np.log10(np.asarray(freqs, float))
        y = np.asarray(mag_dB, float).mean(axis=1)       # 逐频时间均值
        self._lo, self._sc = float(lf.mean()), float(lf.std())
        x = (lf - self._lo) / self._sc

        best = (np.inf, None, None)
        for p in range(1, self.max_order + 1):
            V = np.vander(x, p + 1, increasing=True)
            coef, *_ = np.linalg.lstsq(V, y, rcond=None)
            r = y - V @ coef
            Vp = np.linalg.pinv(V.T @ V)
            h = np.clip(np.einsum('ij,jk,ik->i', V, Vp, V), 0.0, 1 - 1e-12)
            loo = float(np.mean((r / (1.0 - h)) ** 2))
            if loo < best[0]:
                best = (loo, p, coef)

        self.loo_mse, self.order, self.coef = best
        pred = self._design(lf) @ self.coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"[MeanFn] polynomial order {self.order} "
              f"(LOO RMSE {np.sqrt(self.loo_mse):.4f} dB, R^2 {self.r2:.6f}), "
              f"residual per-freq mean |max| = "
              f"{np.abs(y - pred).max():.4f} dB")
        return self

    def __call__(self, freqs):
        """在任意频点（含未观测频点）求值，返回 (N_f,)。"""
        return self._design(np.log10(np.asarray(freqs, float))) @ self.coef

    def residual(self, freqs, mag_dB):
        """返回 mag_dB - m(log10 f)，形状与 mag_dB 相同。"""
        return np.asarray(mag_dB, float) - self(freqs)[:, np.newaxis]

    def restore(self, freqs, resid):
        """把残差域的预测加回均值，返回绝对幅度 [dB]。"""
        return np.asarray(resid, float) + self(freqs)[:, np.newaxis]


# ============================================================
# ②-C 统一指标计算   [revision / R3-6]
# ============================================================

def estimate_measurement_noise(M, axis=1, order=4):
    """
    由数据独立估计 i.i.d. 测量噪声标准差 [dB]。   [revision / R3-7]

    原理：对「平滑信号 + 白噪声」，n 阶差分 D_n 中平滑成分近似抵消，
    而白噪声满足 Var(D_n) = C(2n, n) sigma^2（n=2 → 6，n=4 → 70）。
    用 MAD 做稳健估计。

    [revision, round 2] 默认阶数由 2 改为 4。正弦成分经 n 阶差分后的
    幅度增益为 (4 sin^2(pi f dt))^(n/2)：在 f≈1.6 Hz、dt=50 ms 时，二阶
    约为 0.25，1 dB 的心动振荡就会被误算成约 0.07 dB 的"噪声"；四阶约
    为 0.06，泄漏降低一个数量级。实测 D2/D4/D6 估计分别为
    离体 0.0427/0.0046/0.0012 dB、活体 0.0796/0.0173/0.0073 dB，
    D2 主要反映的是信号曲率。D4 仍是噪声的保守（偏大）估计。

    注意：该估计与显式均值函数 m(log f) 无关——m 不随时间变化，在时间
    方向的差分中自动抵消，因此对原始场和残差场给出同一结果。

    用途：为 WhiteKernel 提供有数据依据的初值与下界，防止短尺度
    Matérn 项把测量噪声吸收掉（否则 noise_level 会一路跑到下界，
    GP 退化为精确插值器，in-sample 指标变得同义反复）。
    """
    from math import comb
    Dn = np.diff(np.asarray(M, float), n=order, axis=axis)
    mad = np.median(np.abs(Dn - np.median(Dn)))
    return float(1.4826 * mad / np.sqrt(comb(2 * order, order)))


CALIB_LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)


def compute_metrics(y_true, y_pred, y_std, label=None):
    """
    统一的指标计算。输入均为 (N_freq, N_time) 的**绝对幅度域** [dB]。

    返回 dict:
        rmse, mae            — 绝对幅度域，与原版本口径一致
        r2                   — 原口径（会被跨频方差主导，仅为与投稿版对照保留）
        r2_demeaned          — 逐频去均值后的 R^2  ★ R3-6 要求的口径
        cov95                — 95% 后验区间经验覆盖率 [%]
        rmse_per_freq        — (N_freq,) 逐频 RMSE   ★ R3-6
        calibration          — {名义水平: 经验覆盖率[%]}  ★ R3-6 校准曲线
    """
    from sklearn.metrics import r2_score

    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ys = np.asarray(y_std, float)

    rmse = float(np.sqrt(mean_squared_error(yt.ravel(), yp.ravel())))
    mae = float(np.mean(np.abs(yt.ravel() - yp.ravel())))
    r2 = float(r2_score(yt.ravel(), yp.ravel()))

    # ── 逐频去均值 R^2：两侧各自减去自己的时间均值 ──────────────────
    # R3-6: "R^2 of 0.998-1.000 is dominated by the tens-of-dB cross-frequency
    #        variance - any model capturing per-frequency means scores near 1"
    ytd = yt - yt.mean(axis=1, keepdims=True)
    ypd = yp - yp.mean(axis=1, keepdims=True)
    r2_dem = float(r2_score(ytd.ravel(), ypd.ravel()))

    err = np.abs(yt - yp)
    cov95 = float(100.0 * np.mean(err <= Z95 * ys))
    calib = {lv: float(100.0 * np.mean(err <= norm.ppf(0.5 + lv / 2.0) * ys))
             for lv in CALIB_LEVELS}
    rmse_pf = np.sqrt(np.mean((yt - yp) ** 2, axis=1))

    out = dict(rmse=rmse, mae=mae, r2=r2, r2_demeaned=r2_dem, cov95=cov95,
               rmse_per_freq=rmse_pf, calibration=calib)
    if label is not None:
        out['label'] = label
    return out


# ============================================================
# ③ SlicedKernel：维度切片包装器
# ============================================================

class SlicedKernel(Kernel):
    """
    将任意 sklearn 核函数约束到输入矩阵的指定列（维度）。

    用途：
        构建只作用于 log10(f)（列 0）或 t（列 1）的子核，
        再通过 sklearn 的 Product/Sum 组合为 2D 乘积核。

    示例:
        k_freq = SlicedKernel(C(1.0) * RBF(0.5), dim=0)  # 只用第0列
        k_time = SlicedKernel(C(1.0) * Matern(), dim=1)   # 只用第1列
        k_2d   = k_freq * k_time                          # 2D 乘积核

    原理：
        __call__(X, Y) 先对 X/Y 取指定列，再调用 base_kernel。
        theta / bounds 属性直接代理到 base_kernel，超参数优化无缝衔接。
    """

    def __init__(self, base_kernel, dim):
        self.base_kernel = base_kernel
        self.dim = int(dim)           # 单列索引（0 = logf，1 = t）

    # ── sklearn 超参数接口（代理到 base_kernel）────────────────────────

    def get_params(self, deep=True):
        params = dict(base_kernel=self.base_kernel, dim=self.dim)
        if deep:
            for k, v in self.base_kernel.get_params(deep=True).items():
                params[f'base_kernel__{k}'] = v
        return params

    @property
    def hyperparameters(self):
        return self.base_kernel.hyperparameters

    @property
    def theta(self):
        return self.base_kernel.theta

    @theta.setter
    def theta(self, theta):
        self.base_kernel.theta = theta

    @property
    def bounds(self):
        return self.base_kernel.bounds

    # ── 核函数计算（仅使用 dim 列）────────────────────────────────────

    def __call__(self, X, Y=None, eval_gradient=False):
        X_s = X[:, [self.dim]]
        Y_s = Y[:, [self.dim]] if Y is not None else None
        return self.base_kernel(X_s, Y_s, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.base_kernel.diag(X[:, [self.dim]])

    def is_stationary(self):
        return self.base_kernel.is_stationary()

    def clone_with_theta(self, theta):
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    def __repr__(self):
        return f"Sliced(dim={self.dim}, {self.base_kernel!r})"


# ============================================================
# ④ 2D 核函数构建
# ============================================================

def build_2d_kernel(heart_period_s=0.833, noise_var=None):
    """
    构建 2D 乘积核（频率轴 × 时间轴）

    结构:
        k_freq  = C_f · RBF(logf)
        k_time  = C_p · ExpSineSquared(t) + C_m · Matérn(t)
        k_2d    = SlicedKernel(k_freq, dim=0)
                × SlicedKernel(k_time, dim=1)
                + WhiteKernel

    输入维度对应:
        dim=0 : log10(f)，范围约 [5.0, 7.0]（100kHz~10MHz）
        dim=1 : t [s]
    """
    # ── 频率轴核（log10 空间，RBF 保证频谱平滑）──────────────────────
    # [revision / Fix A] 删除此处的 ConstantKernel。
    #   原式 C_f*RBF x [C_p*ESS + C_m*Matern] 在变换
    #   (C_f,C_p,C_m) -> (a*C_f, C_p/a, C_m/a) 下逐点不变，似然沿该方向
    #   严格平坦（实测 LML 在 a in [0.1,10] 恒为 20570.9946）。L-BFGS-B
    #   会沿平坦脊滑到盒约束，这正是 C_p 顶下界、C_m 顶上界的原因。
    #   去掉 C_f 后幅度仅由 C_p / C_m 承担，模型可辨识。超参数 8 -> 7。
    k_freq = SlicedKernel(
        RBF(
            length_scale=0.5,              # 初始：半个对数频程
            length_scale_bounds=(0.02, 5.0)   # [Fix C] 0.1 -> 0.02 (≈一个频点间隔)
        ),
        dim=0
    )

    # ── 时间轴核（心动周期性 + 慢变趋势）────────────────────────────
    k_time_periodic = SlicedKernel(
        C(1.0, (1e-2, 1e2)) * ExpSineSquared(
            length_scale=0.15,
            periodicity=heart_period_s,
            length_scale_bounds=(0.05, 20.0),  # [Fix C] 上界 0.5 -> 5.0；[round 3] 5.0 -> 20.0（D4 下活体 ℓ_p 顶在 5，放宽后收敛到内点 6.73）
            periodicity_bounds=(0.4, 2.0)      # 30~150 bpm（维持，有生理依据）
        ),
        dim=1
    )
    # [round 5, 2026-11-09] Matern-3/2 -> RBF. With the noise fixed at the
    #   full-resolution D4 estimate, a consistent kernel ablation (25 f x 80 t,
    #   time-block split) gives ESS + RBF a higher LML than ESS + Matern-3/2
    #   (in vivo +540 nats, ex vivo +102 nats) and a lower time-block RMSE
    #   (0.486 vs 0.514 dB; 0.099 vs 0.106 dB).  The data at 50 ms are smoother
    #   than a once-differentiable process.
    k_time_trend = SlicedKernel(
        C(0.5, (1e-3, 1e2)) * RBF(
            length_scale=2.0,
            length_scale_bounds=(0.05, 20.0)
        ),
        dim=1
    )

    # ── 测量噪声（作用于全空间）──────────────────────────────────────
    # [Fix C] 初值与下界改由数据估计（estimate_measurement_noise）给出。
    #   若放任下界到 1e-8，短尺度 Matérn 会把测量噪声整个吸收，
    #   noise_level 跑到下界、GP 退化为精确插值器（实测 in-sample
    #   RMSE = 0.0000，所有名义水平覆盖率 100%）。
    #   以数据估计值为初值、其 1/10 为下界，既留出优化空间又不致退化。
    #   因此改为：**不优化噪声**，直接固定到 estimate_measurement_noise()
    #   给出的独立估计。这消除了 Matérn/白噪声之间的退化，并把自由
    #   超参数从 8 个降到 6 个（删 C_f 一个、固定噪声一个）。
    if noise_var is None:
        k_noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1.0))
    else:
        k_noise = WhiteKernel(noise_level=float(noise_var),
                              noise_level_bounds="fixed")

    # ── 组合：乘积核 + 噪声核 ────────────────────────────────────────
    #  sklearn 的 Product 分别调用 k_freq 和 k_time_sum，各自只使用自己的列，
    #  结果逐元素相乘，实现 k_freq(logf) × k_time(t) 的乘积分离效果。
    kernel = k_freq * (k_time_periodic + k_time_trend) + k_noise

    return kernel


# ============================================================
# ⑤ 2D 联合 GPR
# ============================================================

class JointGPR2D:
    """
    2D 联合高斯过程回归

    以 (log10(f), t) 为二维输入，训练幅度 GPR 模型。
    相比逐频点 1D GPR，能跨频率共享统计信息，并给出完整的 2D 后验。

    训练数据规模提示:
        训练点数 = N_freq × N_time
        GPR 精确推断复杂度 O(n^3)。
        n ≤ 8000 时精确 GPR 可行（内存 ~500MB，Cholesky ~秒级）；
        n > 8000 时建议使用 subsample 参数降采样或考虑稀疏近似方法。
    """

    def __init__(self, freqs, times, heart_rate_hz=1.2, n_restarts=5):
        self.freqs        = freqs
        self.times        = times
        self.log_freqs    = np.log10(freqs)
        self.heart_period = 1.0 / heart_rate_hz
        self.n_restarts   = n_restarts
        self.N_freq       = len(freqs)
        self.N_time       = len(times)
        self.gpr_mag      = None

    # ── 训练集构建 ──────────────────────────────────────────────────

    def _build_X(self, log_freqs, times):
        """将频率/时间网格展平为 (N_f*N_t, 2) 训练输入矩阵"""
        logf_grid, t_grid = np.meshgrid(log_freqs, times, indexing='ij')
        return np.column_stack([logf_grid.ravel(), t_grid.ravel()])

    # ── 拟合 ────────────────────────────────────────────────────────

    def fit(self, mag_dB, subsample_t=None, subsample_f=None):
        """
        拟合幅度 2D GPR 模型

        参数:
            mag_dB      : (N_freq, N_time) 幅度信道 [dB]
            subsample_t : 时间轴降采样步长（训练点过多时使用，如 2 表示隔一采一）
            subsample_f : 频率轴降采样步长（同上）
        """
        idx_f = np.arange(self.N_freq)[::subsample_f] if subsample_f else np.arange(self.N_freq)
        idx_t = np.arange(self.N_time)[::subsample_t] if subsample_t else np.arange(self.N_time)

        lf_sub  = self.log_freqs[idx_f]
        t_sub   = self.times[idx_t]
        mag_sub = mag_dB[np.ix_(idx_f, idx_t)]

        X_train = self._build_X(lf_sub, t_sub)
        y_mag   = mag_sub.ravel()
        n_train = len(y_mag)

        if n_train > 8000:
            print(f"[Warning] Training points = {n_train} > 8000. "
                  f"Consider fit(..., subsample_t=2) to reduce, "
                  f"or use sparse GPR approximation.")

        print(f"[2D GPR] Training points: {n_train}  "
              f"({len(idx_f)} freq × {len(idx_t)} time)")

        with tqdm(total=1, desc="[2D GPR Fitting]", unit="model",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pbar:
            pbar.set_postfix_str("Fitting magnitude GPR...")
            # [revision] 噪声先验由本次训练数据独立估计（沿时间轴二阶差分）
            _sig = estimate_measurement_noise(mag_sub, axis=1)
            _nv = float((_sig / max(y_mag.std(), 1e-12)) ** 2)
            self.noise_sigma_dB, self.noise_var0 = _sig, _nv
            print(f"[2D GPR] measurement-noise prior: sigma = {_sig:.5f} dB "
                  f"-> noise_level FIXED at {_nv:.3e} (not optimised)")
            # [round 4] coarse-to-fine: dense deterministic multistart on a
            # frequency-subsampled grid, then full-resolution refinement.
            import time as _time
            _t0 = _time.perf_counter()
            _fc = np.arange(len(idx_f))[::COARSE_FREQ_STEP]
            Xc = self._build_X(lf_sub[_fc], t_sub)
            yc = mag_sub[_fc].ravel()
            _gc = GaussianProcessRegressor(
                kernel=build_2d_kernel(self.heart_period, noise_var=_nv),
                n_restarts_optimizer=0, normalize_y=True, alpha=1e-6,
                optimizer=_grid_multistart_lbfgs_optimizer).fit(Xc, yc)
            self.coarse_kernel_ = _gc.kernel_
            print(f"[2D GPR] coarse search ({len(yc)} pts): {_gc.kernel_}  "
                  f"LML={_gc.log_marginal_likelihood_value_:.1f}  ({_time.perf_counter()-_t0:.0f}s)")
            self.gpr_mag = GaussianProcessRegressor(
                kernel=build_2d_kernel(self.heart_period, noise_var=_nv),
                n_restarts_optimizer=0,
                normalize_y=True,
                alpha=1e-6,
                optimizer=_make_refine_optimizer([_gc.kernel_.theta]),
            )
            self.gpr_mag.fit(X_train, y_mag)
            pbar.set_postfix_str(
                f"Done, log marginal likelihood = "
                f"{self.gpr_mag.log_marginal_likelihood_value_:.4f}"
            )
            pbar.update(1)

        print("[2D GPR] Fitting complete.")

    # ── 预测（训练频点 × 给定时刻）──────────────────────────────────

    def predict(self, t_pred):
        """
        在训练频点 × 给定时刻的网格上预测幅度信道特性

        返回（形状均为 (N_freq, N_pred)）:
            mag_pred : 幅度均值  [dB]
            mag_std  : 幅度标准差 [dB]
        """
        return self._predict_on_grid(self.log_freqs, t_pred)

    def predict_grid(self, freqs_pred, t_pred):
        """在任意频率-时间网格上预测（用于高分辨率可视化）"""
        return self._predict_on_grid(np.log10(freqs_pred), t_pred)

    def _predict_on_grid(self, log_freqs_pred, t_pred):
        """内部：按频点逐行推断，显示进度条，返回 (N_lf, N_t) 形状结果。"""
        N_lf, N_tp = len(log_freqs_pred), len(t_pred)
        mu_m_all = np.zeros((N_lf, N_tp))
        s_m_all  = np.zeros((N_lf, N_tp))

        with tqdm(total=N_lf, desc="[2D GPR Predict]", unit="freq",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} freq  "
                              "elapsed:{elapsed}  ETA:{remaining}") as pbar:
            for i, lf in enumerate(log_freqs_pred):
                X_row = np.column_stack([np.full(N_tp, lf), t_pred])
                mu_m, s_m = self.gpr_mag.predict(X_row, return_std=True)
                mu_m_all[i] = mu_m
                s_m_all[i]  = s_m
                pbar.update(1)

        return mu_m_all, s_m_all

    def reconstruct_H(self, t_pred):
        """重建幅度信道矩阵（线性幅度），返回 (N_freq, N_pred)"""
        mag_pred, mag_std = self.predict(t_pred)
        H_pred = 10 ** (mag_pred / 20)
        return H_pred, mag_std

    # ── 核函数表达式输出 ──────────────────────────────────────────────

    @staticmethod
    def _find_periodicity(kernel):
        """
        递归遍历核函数树，定位 ExpSineSquared 的 periodicity 参数。
        搜索路径覆盖 k1 / k2 / base_kernel 三个子节点，
        兼容 SlicedKernel 的嵌套层级。
        """
        if hasattr(kernel, 'periodicity'):
            return kernel.periodicity
        for attr in ('k1', 'k2', 'base_kernel'):
            child = getattr(kernel, attr, None)
            if child is not None:
                result = JointGPR2D._find_periodicity(child)
                if result is not None:
                    return result
        return None

    @staticmethod
    def _find_rbl_lengthscale(kernel):
        """递归查找频率轴 RBF 的 length_scale（位于 dim=0 的 SlicedKernel 中）"""
        if isinstance(kernel, SlicedKernel) and kernel.dim == 0:
            bk = kernel.base_kernel
            # 剥离 ConstantKernel 包装
            if hasattr(bk, 'k2') and hasattr(bk.k2, 'length_scale'):
                return bk.k2.length_scale
            if hasattr(bk, 'length_scale'):
                return bk.length_scale
        for attr in ('k1', 'k2', 'base_kernel'):
            child = getattr(kernel, attr, None)
            if child is not None:
                result = JointGPR2D._find_rbl_lengthscale(child)
                if result is not None:
                    return result
        return None

    def print_kernel_expressions(self):
        """打印幅度 2D GPR 优化后核函数完整表达式及关键超参数。"""
        sep = "=" * 70
        print(f"\n{sep}")
        print("  2D Joint GPR — Magnitude Channel Kernel")
        print(sep)
        print(f"\n  Kernel expression:\n  {self.gpr_mag.kernel_}")
        print(f"\n  Log marginal likelihood: {self.gpr_mag.log_marginal_likelihood_value_:.4f}")

        period_m = self._find_periodicity(self.gpr_mag.kernel_)
        rbl_m    = self._find_rbl_lengthscale(self.gpr_mag.kernel_)
        if period_m is not None:
            print(f"\n  >> Optimised cardiac period : {period_m:.4f} s  "
                  f"≈ {60/period_m:.1f} bpm")
        if rbl_m is not None:
            print(f"  >> Frequency RBF length-scale: {rbl_m:.4f}  "
                  f"(log10 freq units, ≈ {rbl_m:.2f} decades)")
        print(f"\n{sep}")


# ============================================================
# ⑥ 评估函数
# ============================================================

def evaluate(model, mag_dB_true, times):
    """
    Evaluate GPR magnitude prediction at training time points.

    Metrics:
        RMSE  — Root Mean Square Error [dB]
        MAE   — Mean Absolute Error [dB]
        R²    — Coefficient of Determination
        Cov95 — Empirical coverage rate of the ±1.96σ (95%) posterior interval [%]
    """
    from sklearn.metrics import r2_score
    mag_pred, mag_std = model.predict(times)

    y_true = mag_dB_true.ravel()
    y_pred = mag_pred.ravel()

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = np.mean(np.abs(y_true - y_pred))
    r2    = r2_score(y_true, y_pred)
    in_ci = ((mag_dB_true >= mag_pred - Z95 * mag_std) &
             (mag_dB_true <= mag_pred + Z95 * mag_std))
    cov95 = 100 * in_ci.mean()

    print("\n" + "=" * 50)
    print("  Evaluation Results (training time points)")
    print("=" * 50)
    print(f"  RMSE  : {rmse:.4f} dB")
    print(f"  MAE   : {mae:.4f} dB")
    print(f"  R²    : {r2:.4f}")
    print(f"  95%CI Coverage: {cov95:.1f}%  (ideal ≥ 95%)")
    print("=" * 50)

    return dict(rmse=rmse, mae=mae, r2=r2, cov95=cov95)


# ============================================================
# ⑥-B 棋盘式 Holdout 评估
# ============================================================

def checkerboard_holdout(freqs, times, mag_dB,
                         heart_rate_hz=1.2, n_restarts=3,
                         n_holdout_f=16, n_holdout_t=16):
    """
    棋盘式 Train/Test Split 评估。

    2D GPR：从频率轴均匀留出 n_holdout_f 个频点、时间轴均匀留出 n_holdout_t 个
            时间点，组成棋盘空洞区域作为测试集，其余为训练集。
    1D-f GPR (Per-Freq)：留出 n_holdout_t 个时间点。
    1D-t GPR (Per-Time)：留出 n_holdout_f 个频率点。

    返回:
        dict，包含三个模型的 in-sample 与 holdout 指标
    """
    from sklearn.metrics import r2_score

    N_freq, N_time = len(freqs), len(times)

    # ── 均匀选取留出索引 ────────────────────────────────────────────
    holdout_f_idx = np.round(np.linspace(0, N_freq - 1, n_holdout_f + 2)[1:-1]).astype(int)
    holdout_t_idx = np.round(np.linspace(0, N_time - 1, n_holdout_t + 2)[1:-1]).astype(int)
    holdout_f_idx = np.unique(holdout_f_idx)
    holdout_t_idx = np.unique(holdout_t_idx)

    train_f_idx = np.setdiff1d(np.arange(N_freq), holdout_f_idx)
    train_t_idx = np.setdiff1d(np.arange(N_time), holdout_t_idx)

    train_f_mask = np.isin(np.arange(N_freq), train_f_idx)
    train_t_mask = np.isin(np.arange(N_time), train_t_idx)

    print(f"\n{'=' * 70}")
    print(f"  Checkerboard Holdout Evaluation")
    print(f"{'=' * 70}")
    print(f"  Frequency axis: {len(holdout_f_idx)} holdout / {len(train_f_idx)} train "
          f"(total {N_freq})")
    print(f"  Time axis     : {len(holdout_t_idx)} holdout / {len(train_t_idx)} train "
          f"(total {N_time})")

    # ── 构建训练/测试数据 ──────────────────────────────────────────
    # 2D: 训练集 = train_f × train_t; 测试集 = holdout_f × holdout_t (棋盘交叉)
    mag_train_2d = mag_dB[np.ix_(train_f_idx, train_t_idx)]
    mag_test_2d  = mag_dB[np.ix_(holdout_f_idx, holdout_t_idx)]

    freqs_train = freqs[train_f_idx]
    times_train = times[train_t_idx]
    freqs_test  = freqs[holdout_f_idx]
    times_test  = times[holdout_t_idx]

    # ── 指标计算辅助函数 ──────────────────────────────────────────
    # [revision] 改用模块级 compute_metrics，附带 r2_demeaned / 逐频 RMSE / 校准曲线
    def _calc_metrics(y_true, y_pred, y_std):
        return compute_metrics(y_true, y_pred, y_std)

    results = {}

    # [revision / Fix B] 每个方案的均值函数都**只在该方案自己的训练子集上**
    # 拟合，避免留出频点的均值信息泄漏。

    # ── 2D GPR ────────────────────────────────────────────────────
    print("\n  [Holdout] Fitting 2D GPR on training subset...")
    mfn_2d = FrequencyMeanFunction().fit(freqs_train, mag_train_2d)
    model_2d = JointGPR2D(freqs_train, times_train,
                          heart_rate_hz=heart_rate_hz, n_restarts=n_restarts)
    model_2d.fit(mfn_2d.residual(freqs_train, mag_train_2d))

    # In-sample (train freq × train time)
    mp_tr, ms_tr = model_2d.predict(times_train)
    results['2D GPR (train)'] = _calc_metrics(
        mag_train_2d, mfn_2d.restore(freqs_train, mp_tr), ms_tr)

    # Holdout (test freq × test time — 棋盘交叉区域)
    mp_ho, ms_ho = model_2d.predict_grid(freqs_test, times_test)
    results['2D GPR (holdout)'] = _calc_metrics(
        mag_test_2d, mfn_2d.restore(freqs_test, mp_ho), ms_ho)

    # ── Per-Frequency 1D GPR ──────────────────────────────────────
    # 训练：全频点 × 训练时间点；测试：全频点 × 留出时间点
    print("  [Holdout] Fitting Per-Frequency 1D GPR on training time subset...")
    mag_train_pf = mag_dB[:, train_t_idx]
    mfn_pf = FrequencyMeanFunction().fit(freqs, mag_train_pf)
    model_pf = PerFrequencyGPR1D(freqs, times_train,
                                  heart_rate_hz=heart_rate_hz, n_restarts=n_restarts)
    model_pf.fit(mfn_pf.residual(freqs, mag_train_pf))

    mp_pf_tr, ms_pf_tr = model_pf.predict(times_train)
    results['Per-Freq 1D (train)'] = _calc_metrics(
        mag_train_pf, mfn_pf.restore(freqs, mp_pf_tr), ms_pf_tr)

    mp_pf_ho, ms_pf_ho = model_pf.predict(times_test)
    mag_test_pf = mag_dB[:, holdout_t_idx]
    results['Per-Freq 1D (holdout)'] = _calc_metrics(
        mag_test_pf, mfn_pf.restore(freqs, mp_pf_ho), ms_pf_ho)

    # ── Per-Time 1D GPR ───────────────────────────────────────────
    # 训练：训练频点 × 全时间点；测试：留出频点 × 全时间点
    print("  [Holdout] Fitting Per-Time 1D GPR on training freq subset...")
    mag_train_pt = mag_dB[train_f_idx, :]
    mfn_pt = FrequencyMeanFunction().fit(freqs_train, mag_train_pt)
    model_pt = PerTimeGPR1D(freqs_train, times, n_restarts=n_restarts)
    model_pt.fit(mfn_pt.residual(freqs_train, mag_train_pt))

    mp_pt_tr, ms_pt_tr = model_pt.predict(times)
    results['Per-Time 1D (train)'] = _calc_metrics(
        mag_train_pt, mfn_pt.restore(freqs_train, mp_pt_tr), ms_pt_tr)

    mp_pt_ho, ms_pt_ho = model_pt.predict(times, freqs_pred=freqs_test)
    mag_test_pt = mag_dB[holdout_f_idx, :]
    results['Per-Time 1D (holdout)'] = _calc_metrics(
        mag_test_pt, mfn_pt.restore(freqs_test, mp_pt_ho), ms_pt_ho)

    # ── 打印结果汇总表 ────────────────────────────────────────────
    sep = "=" * 92
    print(f"\n{sep}")
    print(f"  {'Method':<28} {'RMSE (dB)':>10} {'MAE (dB)':>10} "
          f"{'R²':>8} {'R²_dem':>9} {'95%CI Cov.':>12}")
    print(sep)
    for label, m in results.items():
        marker = "★" if "holdout" in label else " "
        print(f" {marker} {label:<27} {m['rmse']:>10.4f} {m['mae']:>10.4f} "
              f"{m['r2']:>8.4f} {m['r2_demeaned']:>9.4f} {m['cov95']:>11.1f}%")
    print(sep)
    print("  ★ = holdout (out-of-sample) metrics")
    print("  R²_dem = per-frequency demeaned R² (R3-6 口径)\n")

    return results


# ============================================================
# ⑥-C 连续块 Holdout 评估   [revision / R1-4, R3-6, R4-4, R5-2]
# ============================================================

def block_holdout(freqs, times, mag_dB, heart_rate_hz=1.2, n_restarts=3,
                  n_block_t=20, freq_block=(60, 80), mean_max_order=3):
    """
    连续块留出评估——与棋盘留出（插值）互补，本函数评估**外推**。

    (A) 连续时间块
        训练前 N_t - n_block_t 个时刻，测试最后 n_block_t 个时刻，即
        向前时间外推。默认 n_block_t=20，对 80 点窗口即 60/20 划分，
        与 GRU 的训练/验证划分一致，因此 2D/PF/PT 的结果可直接充当
        GRU 对比所需的「GPR 外推基线」(R3-3)。

    (B) 频段块
        留出一段**内部**连续频段（默认索引 60:80），训练其余频点。
        均值函数 m(·) **仅在训练频点上拟合**，在留出频点上求值，因此
        不存在信息泄漏。使用内部频段而非边缘频段的原因见
        FrequencyMeanFunction 文档中的「已知限制」。
        Per-Freq 1D 在此项下为 N/A——它在未观测频点上根本没有模型，
        这本身就是该方案的结构性局限。

    返回
    ----
    dict，键为 "<模型> (<块类型>)"，值为 compute_metrics 的输出；
    另含 '_meta' 键记录划分索引与均值函数诊断。
    """
    N_freq, N_time = len(freqs), len(times)
    results = {}
    meta = {}

    sep = "=" * 70
    print(f"\n{sep}")
    print("  Block (contiguous) Holdout Evaluation")
    print(sep)

    # ──────────────────────────────────────────────────────────────
    # (A) 连续时间块：前 n_tr 个时刻训练，最后 n_block_t 个时刻测试
    # ──────────────────────────────────────────────────────────────
    n_tr = N_time - n_block_t
    tr_t = np.arange(n_tr)
    te_t = np.arange(n_tr, N_time)
    print(f"\n  [A] Contiguous TIME block: train t[0:{n_tr}] "
          f"({times[0]:.2f}-{times[n_tr-1]:.2f}s), "
          f"test t[{n_tr}:{N_time}] ({times[n_tr]:.2f}-{times[-1]:.2f}s)")

    mag_tr_t = mag_dB[:, tr_t]
    mag_te_t = mag_dB[:, te_t]
    mfn_t = FrequencyMeanFunction(mean_max_order).fit(freqs, mag_tr_t)
    R_tr_t = mfn_t.residual(freqs, mag_tr_t)
    meta['time_block'] = dict(train_idx=tr_t, test_idx=te_t,
                              mean_order=mfn_t.order, mean_r2=mfn_t.r2)

    print("    fitting 2D GPR ...")
    m2 = JointGPR2D(freqs, times[tr_t], heart_rate_hz=heart_rate_hz,
                    n_restarts=n_restarts)
    m2.fit(R_tr_t)
    mp, ms = m2.predict(times[te_t])
    results['2D GPR (time-block)'] = compute_metrics(
        mag_te_t, mfn_t.restore(freqs, mp), ms)

    print("    fitting Per-Freq 1D ...")
    mpf = PerFrequencyGPR1D(freqs, times[tr_t], heart_rate_hz=heart_rate_hz,
                            n_restarts=n_restarts)
    mpf.fit(R_tr_t)
    mp, ms = mpf.predict(times[te_t])
    results['Per-Freq 1D (time-block)'] = compute_metrics(
        mag_te_t, mfn_t.restore(freqs, mp), ms)

    print("    fitting Per-Time 1D ...")
    mpt = PerTimeGPR1D(freqs, times[tr_t], n_restarts=n_restarts)
    mpt.fit(R_tr_t)
    mp, ms = mpt.predict(times[te_t])
    results['Per-Time 1D (time-block)'] = compute_metrics(
        mag_te_t, mfn_t.restore(freqs, mp), ms)

    # ──────────────────────────────────────────────────────────────
    # (B) 频段块：留出内部连续频段
    # ──────────────────────────────────────────────────────────────
    f_te = np.arange(freq_block[0], freq_block[1])
    f_tr = np.setdiff1d(np.arange(N_freq), f_te)
    print(f"\n  [B] Contiguous FREQ block: hold out f[{freq_block[0]}:{freq_block[1]}] "
          f"({freqs[f_te[0]]/1e6:.2f}-{freqs[f_te[-1]]/1e6:.2f} MHz), "
          f"train on {len(f_tr)} freqs")

    mag_ftr = mag_dB[f_tr, :]
    mag_fte = mag_dB[f_te, :]
    mfn_f = FrequencyMeanFunction(mean_max_order).fit(freqs[f_tr], mag_ftr)
    R_ftr = mfn_f.residual(freqs[f_tr], mag_ftr)

    # 均值函数自身在留出频段上的误差（诊断，用于说明 m(·) 外推质量）
    mean_err = mfn_f(freqs[f_te]) - mag_fte.mean(axis=1)
    print(f"    mean-function extrapolation on held-out band: "
          f"RMSE {np.sqrt(np.mean(mean_err**2)):.4f} dB, "
          f"max |err| {np.abs(mean_err).max():.4f} dB")
    meta['freq_block'] = dict(train_idx=f_tr, test_idx=f_te,
                              mean_order=mfn_f.order, mean_r2=mfn_f.r2,
                              mean_extrap_rmse=float(np.sqrt(np.mean(mean_err**2))))

    print("    fitting 2D GPR ...")
    m2f = JointGPR2D(freqs[f_tr], times, heart_rate_hz=heart_rate_hz,
                     n_restarts=n_restarts)
    m2f.fit(R_ftr)
    mp, ms = m2f.predict_grid(freqs[f_te], times)
    results['2D GPR (freq-block)'] = compute_metrics(
        mag_fte, mfn_f.restore(freqs[f_te], mp), ms)

    print("    fitting Per-Time 1D ...")
    mptf = PerTimeGPR1D(freqs[f_tr], times, n_restarts=n_restarts)
    mptf.fit(R_ftr)
    mp, ms = mptf.predict(times, freqs_pred=freqs[f_te])
    results['Per-Time 1D (freq-block)'] = compute_metrics(
        mag_fte, mfn_f.restore(freqs[f_te], mp), ms)

    results['Per-Freq 1D (freq-block)'] = None   # 结构性不适用

    # ── 汇总 ──────────────────────────────────────────────────────
    sep2 = "=" * 92
    print(f"\n{sep2}")
    print(f"  {'Method':<30} {'RMSE (dB)':>10} {'MAE (dB)':>10} "
          f"{'R²':>8} {'R²_dem':>9} {'95%CI Cov.':>12}")
    print(sep2)
    for label, m in results.items():
        if m is None:
            print(f"    {label:<29} {'N/A — no model at unobserved frequencies':>0}")
            continue
        print(f"    {label:<29} {m['rmse']:>10.4f} {m['mae']:>10.4f} "
              f"{m['r2']:>8.4f} {m['r2_demeaned']:>9.4f} {m['cov95']:>11.1f}%")
    print(sep2 + "\n")

    results['_meta'] = meta
    return results


# ============================================================
# ⑦ 可视化
# ============================================================


# ============================================================
# ⑦-B  逐频点 1D GPR（Per-Frequency）
# ============================================================

class PerFrequencyGPR1D:
    """
    逐频点 1D GPR：对每个频点 f_i，以时间 t 为输入独立建模。

    核函数（时间域）:
        k(t,t') = C_p · ExpSineSquared(T_heart)   ← 心动周期性
                + C_m · Matérn(ν=1.5)              ← 慢变趋势
                + WhiteKernel                       ← 测量噪声

    共训练 N_freq 个幅度模型。
    """

    def __init__(self, freqs, times, heart_rate_hz=1.2, n_restarts=5):
        self.freqs        = freqs
        self.times        = times
        self.heart_period = 1.0 / heart_rate_hz
        self.n_restarts   = n_restarts
        self.N_freq       = len(freqs)
        self.gpr_mag_list   = []

    def _time_kernel(self, noise_var=None):
        # [revision / Fix C] 边界与 2D 时间轴核保持一致。
        # 注意：本核是「和」而非「积」，不存在 2D 核那条幅度平坦脊，
        #       C_p / C_m 本身可辨识，故此处保留两个 ConstantKernel。
        k_per = C(1.0, (1e-2, 1e2)) * ExpSineSquared(
            length_scale=0.15, periodicity=self.heart_period,
            length_scale_bounds=(0.05, 20.0),   # [round 3] 与 2D 一致
            periodicity_bounds=(0.4, 2.0)
        )
        k_trend = C(0.5, (1e-3, 1e2)) * RBF(     # [round 5] Matern-3/2 -> RBF, as in 2D
            length_scale=2.0,
            length_scale_bounds=(0.05, 20.0)
        )
        if noise_var is None:
            return k_per + k_trend + WhiteKernel(0.1, (1e-8, 1.0))
        return k_per + k_trend + WhiteKernel(float(noise_var), "fixed")

    def fit(self, mag_dB):
        t_col = self.times.reshape(-1, 1)
        _sig = estimate_measurement_noise(mag_dB, axis=1)
        self.noise_sigma_dB = _sig
        print(f"[Per-Frequency 1D GPR] Fitting {self.N_freq} frequency points... "
              f"(noise prior sigma = {_sig:.5f} dB)")
        with tqdm(total=self.N_freq, desc="  Magnitude", unit="freq",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pb:
            for i in range(self.N_freq):
                gpr = GaussianProcessRegressor(
                    kernel=self._time_kernel(
                        noise_var=float((_sig / max(mag_dB[i, :].std(), 1e-12)) ** 2)),
                    n_restarts_optimizer=0,      # [round 3] same deterministic multistart as 2D
                    normalize_y=True, alpha=1e-6,
                    optimizer=_grid_multistart_lbfgs_optimizer)
                gpr.fit(t_col, mag_dB[i, :])
                self.gpr_mag_list.append(gpr)
                pb.update(1)
        print("[Per-Frequency 1D GPR] Complete.")

    def predict(self, t_pred):
        """返回 (N_freq, N_pred) 的幅度均值和标准差"""
        t_col  = t_pred.reshape(-1, 1)
        N_p    = len(t_pred)
        mag_pred = np.zeros((self.N_freq, N_p))
        mag_std  = np.zeros((self.N_freq, N_p))
        for i in range(self.N_freq):
            mu_m, s_m = self.gpr_mag_list[i].predict(t_col, return_std=True)
            mag_pred[i] = mu_m
            mag_std[i]  = s_m
        return mag_pred, mag_std

    def print_log_marginal_likelihood(self):
        """汇总打印所有子模型的 log marginal likelihood"""
        lmls = [g.log_marginal_likelihood_value_ for g in self.gpr_mag_list]
        total = sum(lmls)
        avg = total / len(lmls)
        print(f"\n  [Per-Freq 1D GPR] Log Marginal Likelihood:")
        print(f"    Sum   : {total:.4f}  (across {len(lmls)} freq models)")
        print(f"    Mean  : {avg:.4f}")
        print(f"    Range : [{min(lmls):.4f}, {max(lmls):.4f}]")
        return total


# ============================================================
# ⑦-C  逐时间点 1D GPR（Per-Time）
# ============================================================

class PerTimeGPR1D:
    """
    逐时间点 1D GPR：对每个时刻 t_j，以 log10(f) 为输入独立建模。

    核函数（频率域）:
        k(logf, logf') = C_f · RBF(l_f)   ← 频谱平滑性
                       + WhiteKernel       ← 测量噪声

    单个时刻下信道关于频率是平滑的，无周期性，故只用 RBF。
    共训练 N_time 个幅度模型。
    """

    def __init__(self, freqs, times, n_restarts=5):
        self.freqs     = freqs
        self.log_freqs = np.log10(freqs)
        self.times     = times
        self.n_restarts = n_restarts
        self.N_time    = len(times)
        self.gpr_mag_list   = []

    def _freq_kernel(self, noise_var=None):
        # [revision / Fix C] 本核亦为「和」，C_f 可辨识，保留。
        return (C(1.0, (1e-2, 1e2)) * RBF(
                    length_scale=0.5,
                    length_scale_bounds=(0.02, 5.0))
                + (WhiteKernel(0.1, (1e-8, 1.0)) if noise_var is None
                   else WhiteKernel(float(noise_var), "fixed")))

    def fit(self, mag_dB):
        lf_col = self.log_freqs.reshape(-1, 1)
        _sig = estimate_measurement_noise(mag_dB, axis=0)   # 沿频率轴
        self.noise_sigma_dB = _sig
        print(f"[Per-Time 1D GPR] Fitting {self.N_time} time points... "
              f"(noise prior sigma = {_sig:.5f} dB)")
        with tqdm(total=self.N_time, desc="  Magnitude", unit="time",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pb:
            for j in range(self.N_time):
                gpr = GaussianProcessRegressor(
                    kernel=self._freq_kernel(
                        noise_var=float((_sig / max(mag_dB[:, j].std(), 1e-12)) ** 2)),
                    n_restarts_optimizer=self.n_restarts,
                    normalize_y=True, alpha=1e-6,
                    random_state=0)              # [round 3] seeded restarts
                gpr.fit(lf_col, mag_dB[:, j])
                self.gpr_mag_list.append(gpr)
                pb.update(1)
        print("[Per-Time 1D GPR] Complete.")

    def predict(self, t_pred, freqs_pred=None):
        """
        在给定时刻网格上预测（最近邻借用策略）。

        参数:
            t_pred    : (N_pred,) 预测时刻
            freqs_pred: (N_fp,)  预测频率，默认使用训练频点

        返回 (N_freq, N_pred):
            mag_pred, mag_std
        """
        if freqs_pred is None:
            freqs_pred = self.freqs
        lf_col = np.log10(freqs_pred).reshape(-1, 1)
        N_fp   = len(freqs_pred)
        N_pred = len(t_pred)

        mag_pred = np.zeros((N_fp, N_pred))
        mag_std  = np.zeros((N_fp, N_pred))

        for k, t in enumerate(t_pred):
            j = int(np.argmin(np.abs(self.times - t)))
            mu_m, s_m = self.gpr_mag_list[j].predict(lf_col, return_std=True)
            mag_pred[:, k] = mu_m
            mag_std[:, k]  = s_m

        return mag_pred, mag_std

    def print_log_marginal_likelihood(self):
        """汇总打印所有子模型的 log marginal likelihood"""
        lmls = [g.log_marginal_likelihood_value_ for g in self.gpr_mag_list]
        total = sum(lmls)
        avg = total / len(lmls)
        print(f"\n  [Per-Time 1D GPR] Log Marginal Likelihood:")
        print(f"    Sum   : {total:.4f}  (across {len(lmls)} time models)")
        print(f"    Mean  : {avg:.4f}")
        print(f"    Range : [{min(lmls):.4f}, {max(lmls):.4f}]")
        return total


# ============================================================
# ⑦-D  三方对比：指标 + 可视化
# ============================================================


# ============================================================
# ⑦-E  三维可视化（散点 + 拟合叠加于同一坐标系）
# ============================================================


# ============================================================
# ⑦-F  3D 合并图（三模型合为一张双栏图）
# ============================================================


# ============================================================
# ⑦-G  离体/活体 2×3 并排对比图
# ============================================================


# ============================================================
# ⑦-H  汇总表输出
# ============================================================


# ============================================================
# ⑦-I  Pipeline 函数
# ============================================================

def run_pipeline(freqs, Ra, Ta, times, heart_rate=1.2, n_restarts=3,
                 n_holdout_f=16, n_holdout_t=16,
                 n_block_t=20, freq_block=(60, 80),
                 save_prefix="result", label="Dataset"):
    """
    运行完整 GPR pipeline：拟合 3 模型 + 评估 + holdout + profiling。

    返回 dict 含所有指标、模型对象、预测结果。
    """
    N_FREQ = len(freqs)
    N_TIME = len(times)
    n_train = N_FREQ * N_TIME

    print(f"\n{'='*60}")
    print(f"  Pipeline: {label}")
    print(f"  N_freq={N_FREQ}, N_time={N_TIME}, n_train={n_train}")
    print(f"{'='*60}")

    # ── 1. Compute channel ──────────────────────────────────────────
    mag_dB = compute_channel(Ra, Ta)

    # ── 1b. 显式均值函数 m(log10 f)  [revision / Fix B] ─────────────
    # 三个模型统一在残差域 ΔH = |H| - m(log10 f) 上拟合，使
    #   (i) Matérn 项不再被迫冒充均值函数（原 C_m 顶上界、ℓ_m > 窗口长度）
    #   (ii) 2D 与 PF 1D 的对比口径对齐（PF 1D 的 normalize_y 原本自带
    #        逐频去均值，2D 只减一个全局标量）
    print(f"\n[{label}] Fitting explicit mean function m(log10 f)...")
    mean_fn = FrequencyMeanFunction().fit(freqs, mag_dB)
    resid_dB = mean_fn.residual(freqs, mag_dB)
    print(f"  [MeanFn] between-frequency variance removed: "
          f"{100*(1-resid_dB.var()/mag_dB.var()):.2f}%  "
          f"(resid std {resid_dB.std():.4f} dB vs original {mag_dB.std():.4f} dB)")

    # ── 2. 2D GPR fitting (with profiling) ──────────────────────────
    print(f"\n[{label}] 2D GPR fitting (on residual)...")
    model_2d = JointGPR2D(freqs, times, heart_rate_hz=heart_rate, n_restarts=n_restarts)
    model_2d.mean_fn = mean_fn          # 供绘图/复现时还原绝对幅度
    tracemalloc.start()
    t0_2d = time.perf_counter()
    model_2d.fit(resid_dB)
    t_2d = time.perf_counter() - t0_2d
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  [Profiling] 2D GPR fit time     : {t_2d:.2f} s")
    print(f"  [Profiling] Peak memory (Python) : {mem_peak / 1e6:.1f} MB")

    # Print kernel
    model_2d.print_kernel_expressions()

    # Evaluate 2D（残差域，自洽口径）
    metrics_2d = evaluate(model_2d, resid_dB, times)

    # ── 3. Per-Frequency 1D GPR ─────────────────────────────────────
    print(f"\n[{label}] Per-Frequency 1D GPR fitting (on residual)...")
    model_pf = PerFrequencyGPR1D(freqs, times, heart_rate_hz=heart_rate, n_restarts=n_restarts)
    model_pf.mean_fn = mean_fn
    t0_pf = time.perf_counter()
    model_pf.fit(resid_dB)
    t_pf = time.perf_counter() - t0_pf
    print(f"  [Profiling] Per-Freq 1D fit time : {t_pf:.2f} s")
    model_pf.print_log_marginal_likelihood()

    # ── 4. Per-Time 1D GPR ──────────────────────────────────────────
    print(f"\n[{label}] Per-Time 1D GPR fitting (on residual)...")
    model_pt = PerTimeGPR1D(freqs, times, n_restarts=n_restarts)
    model_pt.mean_fn = mean_fn
    t0_pt = time.perf_counter()
    model_pt.fit(resid_dB)
    t_pt = time.perf_counter() - t0_pt
    print(f"  [Profiling] Per-Time 1D fit time : {t_pt:.2f} s")
    model_pt.print_log_marginal_likelihood()

    # Training time summary
    print(f"\n  Training Time Summary ({label}):")
    print(f"  2D Joint GPR    : {t_2d:>8.2f} s")
    print(f"  Per-Freq 1D GPR : {t_pf:>8.2f} s")
    print(f"  Per-Time 1D GPR : {t_pt:>8.2f} s")

    # ── 5. High-density prediction（残差域预测后还原为绝对幅度）─────
    t_pred = np.linspace(times[0], times[-1], 400)
    _r2d, mag_std_2d = model_2d.predict(t_pred)
    _rpf, mag_std_pf = model_pf.predict(t_pred)
    _rpt, mag_std_pt = model_pt.predict(t_pred)
    mag_pred_2d = mean_fn.restore(freqs, _r2d)
    mag_pred_pf = mean_fn.restore(freqs, _rpf)
    mag_pred_pt = mean_fn.restore(freqs, _rpt)

    # ── 6. In-sample metrics ────────────────────────────────────────
    _a, ms_2d_tr = model_2d.predict(times); mp_2d_tr = mean_fn.restore(freqs, _a)
    _b, ms_pf_tr = model_pf.predict(times); mp_pf_tr = mean_fn.restore(freqs, _b)
    _c, ms_pt_tr = model_pt.predict(times); mp_pt_tr = mean_fn.restore(freqs, _c)

    compare_results = [
        compute_metrics(mag_dB, mp_2d_tr, ms_2d_tr, label="2D GPR"),
        compute_metrics(mag_dB, mp_pf_tr, ms_pf_tr, label="Per-Freq 1D"),
        compute_metrics(mag_dB, mp_pt_tr, ms_pt_tr, label="Per-Time 1D"),
    ]
    print(f"\n  In-sample comparison ({label}):")
    print(f"  {'Method':<14}{'RMSE (dB)':>11}{'R²':>9}{'R²_dem':>10}{'Cov95':>9}")
    for r in compare_results:
        print(f"  {r['label']:<14}{r['rmse']:>11.4f}{r['r2']:>9.4f}"
              f"{r['r2_demeaned']:>10.4f}{r['cov95']:>8.1f}%")

    # ── 7. Checkerboard holdout（插值）──────────────────────────────
    print(f"\n[{label}] Checkerboard holdout evaluation (interpolation)...")
    holdout_results = checkerboard_holdout(
        freqs, times, mag_dB,
        heart_rate_hz=heart_rate, n_restarts=n_restarts,
        n_holdout_f=n_holdout_f, n_holdout_t=n_holdout_t,
    )

    # ── 7b. Block holdout（外推）[revision]──────────────────────────
    print(f"\n[{label}] Block holdout evaluation (extrapolation)...")
    block_results = block_holdout(
        freqs, times, mag_dB,
        heart_rate_hz=heart_rate, n_restarts=n_restarts,
        n_block_t=n_block_t, freq_block=freq_block,
    )

    # ── 8. Save numpy results ───────────────────────────────────────
    np.save(f"{save_prefix}_mag_dB.npy", mag_dB)
    np.save(f"{save_prefix}_mag_pred_2d.npy", mag_pred_2d)
    np.save(f"{save_prefix}_mag_std_2d.npy", mag_std_2d)
    np.save(f"{save_prefix}_t_pred.npy", t_pred)

    # ── Return all results ──────────────────────────────────────────
    result = {
        'label': label,
        'freqs': freqs, 'times': times, 't_pred': t_pred,
        'n_freq': N_FREQ, 'n_time': N_TIME,
        'mag_dB': mag_dB,
        'models': {'2D': model_2d, 'PF': model_pf, 'PT': model_pt},
        'mag_pred_hd': mag_pred_2d,
        'mag_std_hd': mag_std_2d,
        'preds_hd': {
            '2D GPR':      (mag_pred_2d, mag_std_2d),
            'Per-Freq 1D': (mag_pred_pf, mag_std_pf),
            'Per-Time 1D': (mag_pred_pt, mag_std_pt),
        },
        'compare_results': compare_results,
        'holdout_results': holdout_results,
        'block_results': block_results,          # [revision] 连续块外推
        'mean_fn': mean_fn,                      # [revision] 显式均值函数
        'resid_dB': resid_dB,                    # [revision] GP 实际建模的残差场
        'profiling': {
            't_2d': t_2d, 't_pf': t_pf, 't_pt': t_pt,
            'mem_peak': mem_peak,
        },
    }

    # ── Persist full result for resume ─────────────────────────────
    import pickle
    with open(f"{save_prefix}_result.pkl", "wb") as _f:
        pickle.dump(result, _f)
    print(f"[Pipeline] Full result pickled to {save_prefix}_result.pkl")
    return result


# ============================================================
# ⑧ 文件读取函数
# ============================================================

def _read_file(path, label):
    """
    通用文件读取辅助函数，根据扩展名自动选择 CSV / XLSX 读取方式。
    返回不含标题的纯数值 numpy 数组。
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("读取文件需要 pandas，请先执行: pip install pandas openpyxl")

    ext = str(path).lower().split('.')[-1]
    if ext == 'csv':
        arr = pd.read_csv(path, header=None).values.astype(float)
    elif ext in ('xlsx', 'xls'):
        arr = pd.read_excel(path, header=None, engine='openpyxl').values.astype(float)
    else:
        raise ValueError(f"[{label}] 不支持的文件格式 .{ext}，请使用 .csv / .xlsx")

    print(f"  [{label}] 读取完成，原始 shape = {arr.shape}，路径: {path}")
    return arr


def load_freqs_from_file(path_freqs):
    """
    从单列 XLSX / CSV 文件读取频率数组。

    文件格式约定:
        单列，N_freq 行，每行一个频率值，单位 Hz，无标题行。
        示例（5行单列）:
            100000
            215443
            464159
            1000000
            10000000

    参数:
        path_freqs : 频率文件路径（字符串）

    返回:
        freqs : np.ndarray  (N_freq,)，单位 Hz，升序
    """
    raw = _read_file(path_freqs, "freqs")
    freqs = raw.ravel()

    if not np.all(np.diff(freqs) > 0):
        print("  [freqs] 警告：频率值未严格升序，已自动排序")
        freqs = np.sort(freqs)

    print(f"  [freqs] 频点数 N_freq = {len(freqs)}")
    print(f"  [freqs] 频率范围: {freqs[0]/1e3:.1f} kHz ~ {freqs[-1]/1e6:.3f} MHz")
    return freqs


def load_inputs_from_files(path_Ra, path_Ta, freqs, times):
    """
    从 CSV 或 XLSX 文件读取 Ra、Ta，并校验维度。

    文件格式约定（已知存储方向）
    ─────────────────────────────────────────────────────────
    Ra（时频矩阵）:
        行 = 时刻（N_time 行），列 = 频率（N_freq 列）
        即 Ra[j, i] 为第 j 个时刻、第 i 个频点的接收幅度。
        读入后自动转置为 (N_freq, N_time) 供后续代码使用。

    Ta（频域向量）:
        单列文件，N_freq 行，每行一个数值，单位与 Ra 一致。

    支持格式:  .csv  /  .xlsx  /  .xls
    ─────────────────────────────────────────────────────────

    参数:
        path_Ra : Ra 文件路径
        path_Ta : Ta 文件路径
        freqs   : 频率数组 (N_freq,)，已从文件或参数确定
        times   : 时间数组 (N_time,)，已从参数确定

    返回（均已转置/整形为代码内部约定的维度）:
        Ra : np.ndarray  (N_freq, N_time)
        Ta : np.ndarray  (N_freq,)
    """
    N_freq = len(freqs)
    N_time = len(times)

    print("[文件读取] 开始读取输入文件...")
    Ra_raw = _read_file(path_Ra, "Ra")
    Ta_raw = _read_file(path_Ta, "Ta")

    Ta = Ta_raw.ravel()

    errors = []

    if Ra_raw.shape != (N_time, N_freq):
        errors.append(
            f"Ra 原始 shape={Ra_raw.shape}，"
            f"期望 (N_time={N_time}, N_freq={N_freq})。\n"
            f"     请检查：① 时间参数（T_START/T_END/T_STEP）是否与文件行数一致；"
            f"② 频率文件频点数是否与文件列数一致"
        )

    if len(Ta) != N_freq:
        errors.append(f"Ta 长度={len(Ta)}，期望 N_freq={N_freq}")

    if errors:
        msg = "\n  ✗ ".join(errors)
        raise ValueError(f"[文件读取] 维度校验失败:\n  ✗ {msg}")

    Ra = Ra_raw.T

    print(f"[文件读取] 维度校验通过 ✓  （已将矩阵从行=时刻/列=频率转置为行=频率/列=时刻）")
    print(f"  Ra : 文件 {Ra_raw.shape} → 转置后 {Ra.shape}  (N_freq × N_time)")
    print(f"  Ta : {Ta.shape}")

    return Ra, Ta


# ============================================================
# ⑨ 主程序 — 离体 / 活体双数据集对比
# ============================================================

