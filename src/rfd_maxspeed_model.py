"""
RFD-limited impulse model of top sprint speed.

三つの量 —— 筋力の立ち上がり率 (rate of force development, RFD, R [N/s]) ／
接地期の鉛直床反力ピーク (peak GRF, F_peak) ／ 最大疾走速度 (v_max) —— を
最小限の第一原理モデルで結ぶ。

物理的枠組み (Weyand et al. 2000, 2010 の力積フレームワーク):

  定常最高速度での周期走行では、1ステップ (接地 t_c + 滞空 t_a) にわたる
  鉛直方向の正味力積がゼロ。ゆえに接地期の鉛直GRF力積が全体重を支える:

        J_grf = ∫_0^{t_c} F_v dt = m g (t_c + t_a)                     (1)

  接地長 L_c (接地中に身体重心が進む水平距離) を一定とみなすと

        t_c = L_c / v                                                  (2)

  有限のRFD (R) は接地時間内で到達できる力の立ち上がりを制限する。接地波形を
  「立ち上がり率 R で上昇し、生理学的上限 F_max で頭打ち、対称に下降」する
  三角形/台形波としてモデル化すると、接地時間 t_c で到達可能な最大力積は

        rise time to cap: t_rise = F_max / R
        t_c/2 <= t_rise :  三角形  J_avail = (1/4) R t_c^2,  F_peak = R t_c/2
        t_c/2 >  t_rise :  台形    J_avail = F_max t_c - F_max^2/R, F_peak = F_max  (3)

  最高速度 v_max は「必要力積(式1) = 到達可能力積(式3)」を満たす最大の v。

解析的帰結:
  - RFD律速域 (三角形): (1/4) R t_c^2 = m g (t_c + t_a) を t_c=L_c/v で解くと
    v_max(R) の閉形式が得られ、R が大きい極限で v_max ∝ sqrt(R)。
  - 筋力(F_max)律速域: R→∞ で J_avail→F_max t_c (矩形)、v_max は F_max で決まる。
  - この2域の遷移として最高速度が決まる。

このスクリプトは (A) 解析閉形式、(B) 波形を数値積分して求めた v_max、
(C) sympy による閉形式の独立検証、を相互照合し、スケーリング則と遷移を図示する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class SprintParams:
    """
    標準パラメータ。既定値は三角形波 (beta=0.5) の説明用イラスト値。
    実測較正値は src/calibrate_parameters.py で構築する。

    波形モデル: 立ち上がり率 R で上昇し F_max で頭打ちする対称台形波。
    beta = (片側の立ち上がり時間)/t_c  (0<beta<=0.5)。
      - beta=0.5 : 平坦部なしの三角形 (形状因子 kappa=F_peak/F_avg=2)
      - beta<0.5 : 平坦部ありの台形 (kappa=1/(1-beta)<2, より「充実」した波形)
    """

    m: float = 80.0        # 体重 [kg]
    g: float = 9.81        # 重力加速度 [m/s^2]
    L_c: float = 1.0       # 接地長 (接地中の重心水平移動) [m]
    t_a: float = 0.12      # 滞空時間 [s] (Weyand: 速度によらずほぼ一定)
    F_max: float = 6000.0  # 鉛直GRFの生理学的上限 [N] (~7.6 body weight)
    beta: float = 0.5      # 立ち上がり時間割合 (0.5=三角形)

    @property
    def W(self) -> float:
        return self.m * self.g

    @property
    def kappa(self) -> float:
        """波形形状因子 F_peak/F_avg = 1/(1-beta)。"""
        return 1.0 / (1.0 - self.beta)


# --------------------------------------------------------------------------
# (3) 到達可能力積・ピーク力  (三角形/台形波、rate R・cap F_max)
# --------------------------------------------------------------------------
def available_impulse(t_c: float, R: float, p: SprintParams) -> float:
    """接地時間 t_c で到達可能な最大鉛直力積 J_avail [N·s]（対称台形波）。"""
    b = p.beta
    if R * b * t_c <= p.F_max:            # rate-limited: ピークが cap に届かない
        return b * (1.0 - b) * R * t_c**2
    return p.F_max * t_c - p.F_max**2 / R  # strength-limited: 立ち上がりは R が律速


def peak_force(t_c: float, R: float, p: SprintParams) -> float:
    """接地時間 t_c での鉛直GRFピーク F_peak [N]。"""
    return min(p.F_max, R * p.beta * t_c)


def required_impulse(t_c: float, p: SprintParams) -> float:
    """式(1): 全ステップの体重を支えるのに必要な鉛直力積 [N·s]。"""
    return p.W * (t_c + p.t_a)


# --------------------------------------------------------------------------
# (B) 数値: 波形の力積収支を根探索して v_max を求める
# --------------------------------------------------------------------------
def impulse_residual(v: float, R: float, p: SprintParams) -> float:
    """Phi(v) = J_avail - J_req。v_max はこの符号反転点。"""
    t_c = p.L_c / v
    return available_impulse(t_c, R, p) - required_impulse(t_c, p)


def v_max_numerical(R: float, p: SprintParams,
                    v_lo: float = 1.0, v_hi: float = 80.0) -> float:
    """力積収支の根として v_max を数値的に求める (scipy.brentq)。"""
    # Phi は v の減少関数。v_lo で正 (実行可能)、v_hi で負 (不可能) を確認。
    if impulse_residual(v_lo, R, p) <= 0:
        raise ValueError("v_lo が既に実行不可能。v_lo を下げてください。")
    if impulse_residual(v_hi, R, p) >= 0:
        raise ValueError("v_hi でもまだ実行可能。v_hi を上げてください。")
    return brentq(impulse_residual, v_lo, v_hi, args=(R, p), xtol=1e-12, rtol=1e-14)


# --------------------------------------------------------------------------
# (A) 解析: 閉形式 v_max(R)
# --------------------------------------------------------------------------
def v_max_analytic(R: float, p: SprintParams) -> tuple[float, str]:
    """
    閉形式で v_max(R) を返す。RFD律速(三角形)域と筋力律速(台形)域の
    両方を計算し、実際に自分の regime 条件を満たす解を採用する。
    戻り値 (v_max, regime)。
    """
    W, L_c, t_a, F_max, bt = p.W, p.L_c, p.t_a, p.F_max, p.beta

    # --- rate-limited 域: beta(1-beta) R (L_c/v)^2 = W (L_c/v + t_a) ---
    # x = 1/v とおくと  beta(1-beta) R L_c^2 x^2 - W L_c x - W t_a = 0
    a = bt * (1.0 - bt) * R * L_c**2
    b = -W * L_c
    c = -W * t_a
    x_tri = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)   # 正の根
    v_tri = 1.0 / x_tri
    t_c_tri = L_c / v_tri

    # --- strength-limited 域: F_max t_c - F_max^2/R = W (t_c + t_a) ---
    #   t_c (F_max - W) = F_max^2/R + W t_a
    if F_max > W:
        t_c_trap = (F_max**2 / R + W * t_a) / (F_max - W)
        v_trap = L_c / t_c_trap
    else:
        t_c_trap, v_trap = np.inf, 0.0

    # regime 判定: R*beta*t_c <= F_max なら rate-limited が binding
    if R * bt * t_c_tri <= F_max:
        return v_tri, "rate-limited"
    return v_trap, "strength-limited"


# --------------------------------------------------------------------------
# 実行: 相互検証 + スケーリング + 図
# --------------------------------------------------------------------------
def sympy_verify_triangular() -> str:
    """三角形域の閉形式根を sympy で独立に導出・照合する。"""
    import sympy as sp

    R, L_c, W, t_a, v = sp.symbols("R L_c W t_a v", positive=True)
    # (1/4) R (L_c/v)^2 = W (L_c/v + t_a)
    eq = sp.Eq(sp.Rational(1, 4) * R * (L_c / v) ** 2, W * (L_c / v + t_a))
    sols = sp.solve(eq, v)
    sols_pos = [s for s in sols if sp.simplify(s) != 0]
    # 大きい R での漸近: sqrt(R) スケーリングを確認
    v_sol = [s for s in sols if s.is_real is not False]
    asymptote = sp.limit(sols_pos[0] / sp.sqrt(R), R, sp.oo) if sols_pos else None
    return f"sympy roots: {sols_pos}\n  v_max/sqrt(R) as R->inf : {asymptote}"


def main() -> None:
    p = SprintParams()

    print("=" * 70)
    print("RFD-limited impulse model of top sprint speed")
    print("=" * 70)
    print(f"params: m={p.m} kg, g={p.g}, W={p.W:.1f} N, "
          f"L_c={p.L_c} m, t_a={p.t_a} s, F_max={p.F_max} N ({p.F_max/p.W:.1f} BW)")

    # --- (A) vs (B): 閉形式と数値の照合 ---
    print("\n[相互検証] 解析閉形式 vs 数値根探索  (v_max, m/s)")
    print(f"{'R [kN/s]':>10} {'v_analytic':>12} {'v_numeric':>12} "
          f"{'rel.err':>10} {'regime':>28}")
    R_list = [20e3, 40e3, 60e3, 86e3, 120e3, 200e3, 400e3, 1e6]
    max_rel_err = 0.0
    for R in R_list:
        v_a, regime = v_max_analytic(R, p)
        v_n = v_max_numerical(R, p)
        rel = abs(v_a - v_n) / v_n
        max_rel_err = max(max_rel_err, rel)
        print(f"{R/1e3:>10.0f} {v_a:>12.5f} {v_n:>12.5f} {rel:>10.1e} "
              f"{regime:>28}")
    print(f"  --> 最大相対誤差 = {max_rel_err:.2e}  "
          f"({'PASS' if max_rel_err < 1e-6 else 'FAIL'}: 解析=数値)")

    # --- ピークGRFと接地時間 (代表点: R=86 kN/s) ---
    R0 = 86e3
    v0, reg0 = v_max_analytic(R0, p)
    t_c0 = p.L_c / v0
    Fp0 = peak_force(t_c0, R0, p)
    print(f"\n[代表点 R={R0/1e3:.0f} kN/s] "
          f"v_max={v0:.2f} m/s, t_c={t_c0*1e3:.0f} ms, "
          f"F_peak={Fp0:.0f} N ({Fp0/p.W:.1f} BW), regime={reg0}")

    # --- (C) sympy 独立検証 ---
    print("\n[sympy 独立検証] 三角形域の閉形式根")
    print("  " + sympy_verify_triangular().replace("\n", "\n  "))

    # --- スケーリング指数: log-log 傾き ---
    # 三角形分枝内で s は 低RFDで 1 (線形: t_c>>t_a) → 高RFDで 0.5 (√R: t_c<<t_a)
    # と減少し、筋力(F_max)律速に入ると s→0 (頭打ち)。この3領域を確認する。
    print("\n[スケーリング則] v_max ∝ R^s の局所指数 s = d ln v_max / d ln R")
    print("  (三角形分枝: 低RFD s→1 [線形] → 高RFD s→0.5 [√R]; 筋力律速: s→0)")
    R_fine = np.logspace(np.log10(2e3), np.log10(3e6), 400)
    v_fine = np.array([v_max_analytic(R, p)[0] for R in R_fine])
    slope = np.gradient(np.log(v_fine), np.log(R_fine))
    for label, R_q in [("超低RFD", 3e3), ("低RFD", 20e3), ("作動点", 86e3),
                       ("高RFD", 400e3), ("超高RFD", 3e6)]:
        s_q = float(np.interp(np.log(R_q), np.log(R_fine), slope))
        _, reg_q = v_max_analytic(R_q, p)
        print(f"  {label:>7} (R={R_q/1e3:>6.0f} kN/s): s={s_q:+.3f}   [{reg_q}]")

    # --- 図 ---
    _make_figure(p, R_fine, v_fine, slope)
    print("\n図を output/rfd_maxspeed_scaling_illustrative.png に保存しました"
          "（説明用の既定値。原稿の Fig. 1 は src/figure_scaling_law.py が生成）。")


def _make_figure(p: SprintParams, R_fine, v_fine, slope) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 現実の作動点 (R=86 kN/s)
    R_op = 86e3
    v_op, _ = v_max_analytic(R_op, p)

    # (1) v_max vs RFD (log-log) + sqrt(R) 参照線
    ax = axes[0]
    ax.loglog(R_fine / 1e3, v_fine, lw=2, label="model $v_{max}(R)$")
    # sqrt(R) 参照線は √R 域 (t_c<<t_a) の漸近定数で規格化
    ref = (p.L_c / (2 * np.sqrt(p.W * p.t_a))) * np.sqrt(R_fine)
    ax.loglog(R_fine / 1e3, ref, "--", color="gray",
              label=r"$\propto\sqrt{R}$ asymptote")
    ax.axhline(v_fine[-1], ls=":", color="crimson",
               label="strength-limited plateau")
    ax.plot(R_op / 1e3, v_op, "o", color="k", ms=7, zorder=5,
            label=f"elite operating point\n({v_op:.1f} m/s)")
    ax.set_xlabel("RFD  R  [kN/s]")
    ax.set_ylabel(r"top speed  $v_{max}$  [m/s]")
    ax.set_title("(a) max speed vs RFD")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (2) 局所スケーリング指数
    ax = axes[1]
    ax.semilogx(R_fine / 1e3, slope, lw=2, color="C2")
    ax.axhline(0.5, ls="--", color="gray")
    ax.axhline(0.0, ls=":", color="crimson")
    ax.set_xlabel("RFD  R  [kN/s]")
    ax.set_ylabel(r"local exponent  $s=d\ln v_{max}/d\ln R$")
    ax.set_title(r"(b) exponent: 1 (linear) $\to$ 0.5 ($\sqrt{R}$) $\to$ 0 (strength)")
    ax.grid(True, which="both", alpha=0.3)

    # (3) ピークGRF vs 速度 (要求ピーク)
    ax = axes[2]
    v_grid = np.linspace(6.0, 26.0, 200)   # 現実的なスプリント速度域
    t_c = p.L_c / v_grid
    # 各 v で力積収支を満たす三角形波の要求ピーク: F_peak = 2 J_req / t_c
    Fp_req = 2.0 * required_impulse(t_c, p) / t_c
    ax.plot(v_grid, Fp_req / p.W, lw=2, color="C3", label="required (triangular)")
    ax.axhline(p.F_max / p.W, ls=":", color="crimson", label="$F_{max}$ ceiling")
    ax.axvline(v_op, ls="--", color="k", alpha=0.6,
               label=f"operating point {v_op:.1f} m/s")
    ax.set_xlabel("running speed  v  [m/s]")
    ax.set_ylabel(r"required peak GRF  $F_{peak}/W$  [body weights]")
    ax.set_title("(c) required peak GRF grows with speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    # NOTE: illustrative defaults, not the calibrated parameters. The
    # manuscript figure is written by src/figure_scaling_law.py; keep the
    # filenames distinct so this demo cannot overwrite it.
    fig.savefig(os.path.join(out_dir, "rfd_maxspeed_scaling_illustrative.png"),
                dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
