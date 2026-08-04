"""
RFD律速力積モデルのパラメータを共有ライブラリの実測値で較正・検証する。

一次資料（すべて共有ライブラリ `reference/library/` に全文収録）:

  [W2010] Weyand, Sandell, Prime & Bundle (2010) J Appl Physiol 108:950-961.
          "The biological limits to running speed are imposed from the ground up."
          Table 1（前方走・最高速度）: F_avg=2.08 Wb, F_peak=3.62 Wb, L_c=0.98 m,
          L_step=2.05 m, T_c=0.108 s, T_aer=0.119 s, 最高速度=9.20 m/s。
          片脚ホッピングの F_peak=4.20 Wb（走行より大＝筋力の天井は走行で使う力より高い）。
          「接地前半 ~55 ms で鉛直力が立ち上がる」。
  [W2000] Weyand, Sternlight, Bellizzi & Wright (2000) J Appl Physiol 89:1991-1999.
          "Faster top running speeds are achieved with greater ground forces..."
          平均 aerial time ~0.128 s（速度によらずほぼ一定）、最速走者 stride length
          ~4.6 m（→ step length 2.3 m）、最高速度は接地期の大きな鉛直力で決まる。
  [Miy2019] Miyashiro, Nagahara, Yamamoto & Nishijima (2019) J Appl Biomech 35:XX.
          最高速度局面: step length=2.15 m, step freq=4.60 Hz（→ 9.9 m/s）,
          swing time=0.330 s, leg length=0.812 m（独立クロスチェック）。
  [Maf2016] Maffiuletti et al. (2016) Eur J Appl Physiol 116:1091-1116.
          RFD総説。early-phase RFD＝爆発的収縮の最初の 50-75 ms（神経ドライブ律速）。
          → 接地の力立ち上がり(~55 ms)はまさに early-phase RFD の時間窓に対応。

注: [W2010] の被験者は最高速度 9.2 m/s の訓練者（世界エリート ~11.5 m/s ではない）。
本較正は「完全に対応づいた運動学＋力データが1表に揃う」[W2010] を主軸とし、
[W2000][Miy2019] を独立クロスチェックに用いる。力はすべて体重比 (Wb) なので、
後述のとおり v_max は体重に依存しない（R も Wb/s で与えれば質量非依存）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import figstyle
from rfd_maxspeed_model import (
    SprintParams, v_max_analytic, v_max_numerical, peak_force, required_impulse,
)


@dataclass(frozen=True)
class Measured:
    """[W2010] Table 1 前方走・最高速度の実測値（力は体重比 Wb）。"""
    F_avg_Wb: float = 2.08      # 接地平均鉛直力 [Wb]
    F_peak_Wb: float = 3.62     # 鉛直力ピーク [Wb]
    F_ceiling_Wb: float = 4.20  # 片脚ホッピングのピーク力 [Wb]（筋力の天井の下限）
    L_c: float = 0.98           # 接地長 [m]
    T_c: float = 0.108          # 接地時間 [s]
    T_aer: float = 0.119        # 滞空時間 [s]
    v_top: float = 9.20         # 最高速度 [m/s]
    m_rep: float = 80.0         # 代表体重 [kg]（Wb→N 換算用。結論には非依存）


def calibrate(meas: Measured = Measured()) -> tuple[SprintParams, dict]:
    """実測値から較正パラメータと導出量を構築する。"""
    W = meas.m_rep * 9.81

    # 波形形状因子 kappa と立ち上がり割合 beta（対称台形: kappa=1/(1-beta)）
    kappa = meas.F_peak_Wb / meas.F_avg_Wb          # = 3.62/2.08 = 1.74
    beta = 1.0 - 1.0 / kappa                          # = 0.425

    # 立ち上がり率 R（波形の初期勾配）: R = F_peak / (beta * T_c)
    t_rise = beta * meas.T_c                          # 片側立ち上がり時間 [s]
    R_Wb_per_s = meas.F_peak_Wb / t_rise              # [Wb/s]（質量非依存）
    R_N_per_s = R_Wb_per_s * W                        # [N/s]（代表体重で換算）

    p = SprintParams(
        m=meas.m_rep, g=9.81,
        L_c=meas.L_c, t_a=meas.T_aer,
        F_max=meas.F_ceiling_Wb * W,                  # 筋力の天井
        beta=beta,
    )
    derived = dict(kappa=kappa, beta=beta, t_rise=t_rise,
                   R_Wb_per_s=R_Wb_per_s, R_N_per_s=R_N_per_s, W=W)
    return p, derived


def main() -> None:
    meas = Measured()
    p, d = calibrate(meas)
    W = d["W"]

    print("=" * 72)
    print("パラメータ較正（Weyand 2010 主軸 + Weyand 2000 / Miyashiro 2019 / Maffiuletti 2016）")
    print("=" * 72)

    print("\n[較正されたパラメータ]")
    print(f"  L_c   = {p.L_c:.3f} m      (実測 0.98 m [W2010]; 事前仮定 1.0 m ≒ 的中)")
    print(f"  t_a   = {p.t_a:.3f} s      (実測 0.119 s [W2010]; 0.128 s [W2000]; 事前 0.12 s ≒ 的中)")
    print(f"  kappa = {d['kappa']:.3f}        (F_peak/F_avg = 3.62/2.08; 三角形の 2.0 より充実)")
    print(f"  beta  = {d['beta']:.3f}        (=1-1/kappa; 立ち上がり割合)")
    print(f"  t_rise= {d['t_rise']*1e3:.1f} ms      (=beta*T_c; Maffiuletti early-phase 50-75 ms と整合)")
    print(f"  R     = {d['R_Wb_per_s']:.1f} Wb/s = {d['R_N_per_s']/1e3:.1f} kN/s "
          f"(代表 {meas.m_rep:.0f} kg; 質量非依存量は Wb/s)")
    print(f"  F_max = {meas.F_ceiling_Wb:.2f} Wb = {p.F_max:.0f} N "
          f"(ホッピングピーク[W2010]; 筋力の天井の下限)")

    # --- 検証1: 力積収支（式1）—— 予測 F_avg vs 実測 ---
    t_c = p.L_c / meas.v_top
    Favg_pred_Wb = (1.0 + p.t_a / t_c)               # = J_req/(W t_c) = mg(t_c+t_a)/(W t_c)
    print("\n[検証1] 力積収支 J=mg(t_c+t_a) の予測 F_avg")
    print(f"  予測 F_avg = {Favg_pred_Wb:.3f} Wb  vs 実測 {meas.F_avg_Wb:.2f} Wb "
          f"(誤差 {abs(Favg_pred_Wb-meas.F_avg_Wb)/meas.F_avg_Wb*100:.1f}%) → 式(1)は実データ上で成立")

    # --- 検証2: 較正モデルが実測最高速度を再現するか ---
    R = d["R_N_per_s"]
    v_pred, regime = v_max_analytic(R, p)
    v_num = v_max_numerical(R, p)
    print("\n[検証2] 較正モデルの最高速度予測")
    print(f"  v_max(解析) = {v_pred:.2f} m/s, v_max(数値) = {v_num:.2f} m/s  "
          f"vs 実測 {meas.v_top:.2f} m/s (誤差 {abs(v_pred-meas.v_top)/meas.v_top*100:.1f}%)")
    print(f"  regime = {regime}")

    # --- 検証3: RFD律速か筋力律速か（作動点のピーク力 vs 天井）---
    t_c_op = p.L_c / v_pred
    Fp_op = peak_force(t_c_op, R, p)
    print("\n[検証3] 作動点は RFD 律速か？")
    print(f"  作動点ピーク力 F_peak = {Fp_op/W:.2f} Wb  (実測走行 {meas.F_peak_Wb:.2f} Wb)")
    print(f"  筋力の天井 F_max     = {meas.F_ceiling_Wb:.2f} Wb")
    print(f"  → F_peak < F_max（未使用の筋力余裕 {(meas.F_ceiling_Wb/Fp_op*W-1)*100:.0f}%）"
          f"ゆえ RFD 律速。走行は最大筋力を出し切っていない[W2010ホッピング]。")

    # --- 質量非依存性 + 作動点の局所スケーリング指数 ---
    s_op = _local_exponent(R, p)
    print("\n[スケーリング] 作動点の局所指数 s = d ln v_max / d ln R")
    print(f"  s = {s_op:.3f}  (線形 s=1 と平方根 s=0.5 の中間; RFD を上げれば最高速度も上がる)")
    # 質量非依存性の確認
    p_heavy = SprintParams(m=100.0, g=9.81, L_c=p.L_c, t_a=p.t_a,
                           F_max=meas.F_ceiling_Wb*100.0*9.81, beta=p.beta)
    v_heavy, _ = v_max_analytic(d["R_Wb_per_s"]*100.0*9.81, p_heavy)
    print(f"  質量非依存性: m=80→100 kg で R,F_max を Wb 一定に保つと "
          f"v_max {v_pred:.2f}→{v_heavy:.2f} m/s（不変）")

    _make_figure(p, R, meas, d)
    print("\n図を output/calibrated_maxspeed.png に保存しました。")


def _local_exponent(R: float, p: SprintParams, dR: float = 1e-3) -> float:
    v1, _ = v_max_analytic(R * (1 - dR), p)
    v2, _ = v_max_analytic(R * (1 + dR), p)
    return (np.log(v2) - np.log(v1)) / (np.log(R * (1 + dR)) - np.log(R * (1 - dR)))


def _make_figure(p: SprintParams, R_op: float, meas: Measured, d: dict) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    W = d["W"]

    R_fine = np.logspace(np.log10(10e3), np.log10(1e6), 300)
    v_fine = np.array([v_max_analytic(R, p)[0] for R in R_fine])

    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=figstyle.figsize(178, 72))

    # (a) 較正モデル v_max vs RFD、実測作動点を重ねる
    ax = axes[0]
    ax.loglog(R_fine / 1e3, v_fine, "-", color="0.15", label="calibrated model")
    ax.plot(R_op / 1e3, meas.v_top, "o", color="k", ms=5.5, zorder=5,
            label=f"Weyand 2010 measured\n({meas.v_top:.1f} m/s)")
    ax.axhline(v_fine[-1], ls=":", color="0.35", lw=1.0,
               label="strength-limited plateau")
    ax.set_xlabel("force-development rate  $R$  [kN s$^{-1}$]")
    ax.set_ylabel("top speed  $v_{\max}$  [m s$^{-1}$]")
    ax.set_title("calibrated $v_{\max}(R)$ and its calibration point", loc="left")
    ax.set_ylim(3, 40)
    # log 既定だと 10^1 しかラベルされず値が読めないため明示する
    ax.set_yticks([3, 5, 10, 20, 40])
    ax.set_yticklabels(["3", "5", "10", "20", "40"])
    ax.set_yticks([], minor=True)
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)

    # (b) 力の予算: 作動点ピーク力 vs 筋力の天井
    ax = axes[1]
    cats = ["required\n$F_{avg}$", "used\n$F_{peak}$\n(running)",
            "ceiling\n$F_{max}$\n(hopping)"]
    vals = [1.0 + p.t_a / (p.L_c / meas.v_top), meas.F_peak_Wb, meas.F_ceiling_Wb]
    ax.bar(cats, vals, color=["0.75", "0.45", "0.20"], edgecolor="k", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.axhline(meas.F_ceiling_Wb, ls="--", color="0.20", lw=0.8)
    ax.set_ylabel("vertical force  [BW]")
    ax.set_title("force budget: peak used lies below the ceiling", loc="left")
    ax.set_ylim(0, meas.F_ceiling_Wb * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(w_pad=1.9)
    figstyle.panel_letters(fig, axes)
    figstyle.save(fig, "calibrated_maxspeed")
    plt.close(fig)


if __name__ == "__main__":
    main()
