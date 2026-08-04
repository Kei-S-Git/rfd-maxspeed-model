"""
Fig. 1（トップスピード則）を**較正パラメータ**で生成する。

`src/rfd_maxspeed_model.py` の `_make_figure` は SprintParams の既定値
（説明用イラスト値: F_max = 6000 N = 7.65 BW, beta = 0.5）で描いており、
作動点マーカーも R = 86 kN/s をハードコードしていた。その結果、投稿版の
Fig. 1 は本文の較正値（天井 4.20 BW、作動点 61.8 kN/s → 9.01 m/s）と
矛盾し、キャプションの「マーカーは較正された作動点」は事実に反していた。

本スクリプトは較正値のみを用いて Fig. 1 を再生成する。旧パネル (c)
（要求ピーク GRF vs 速度、5-26 m/s）は Fig. 2b・Fig. 3c と重複し、かつ
半分以上が人間の到達しない速度域だったため落とした。

実行: python src/figure_scaling_law.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import figstyle
from calibrate_parameters import Measured, calibrate
from rfd_maxspeed_model import v_max_analytic

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def make_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figstyle.apply()

    meas = Measured()
    p, d = calibrate(meas)
    R_op = d["R_N_per_s"]                       # 較正された作動点 [N/s]
    v_op, regime_op = v_max_analytic(R_op, p)

    R = np.logspace(np.log10(5e3), np.log10(5e6), 600)
    v = np.array([v_max_analytic(r, p)[0] for r in R])
    s = np.gradient(np.log(v), np.log(R))

    fig, axes = plt.subplots(1, 2, figsize=figstyle.figsize(178, 72))

    # ---- (a) top-speed law --------------------------------------------------
    ax = axes[0]
    ax.loglog(R / 1e3, v, "-", color="0.15", label="model  $v_{\\max}(R)$")
    ref = (p.L_c / (2 * np.sqrt(p.W * p.t_a))) * np.sqrt(R)
    ax.loglog(R / 1e3, ref, "--", color="0.55", lw=1.0,
              label="$\\propto\\sqrt{R}$ asymptote")
    ax.axhline(v[-1], ls=":", color="0.35", lw=1.0,
               label=f"strength-limited plateau\n($F_{{\\max}}$ = "
                     f"{meas.F_ceiling_Wb:.2f} BW)")
    ax.plot(R_op / 1e3, v_op, "o", color="k", ms=5.5, zorder=5,
            label=f"calibrated operating point\n({R_op/1e3:.1f} kN s$^{{-1}}$, "
                  f"{v_op:.2f} m s$^{{-1}}$)")
    ax.set_xlabel("force-development rate  $R$  [kN s$^{-1}$]")
    ax.set_ylabel("top speed  $v_{\\max}$  [m s$^{-1}$]")
    ax.set_title("top-speed law", loc="left")
    ax.set_ylim(3, 40)
    # 既定の log 目盛だと 10^1 しかラベルされず値が読めないため明示する
    ax.set_yticks([3, 5, 10, 20, 40])
    ax.set_yticklabels(["3", "5", "10", "20", "40"])
    ax.set_yticks([], minor=True)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)

    # ---- (b) local exponent -------------------------------------------------
    ax = axes[1]
    ax.semilogx(R / 1e3, s, "-", color="0.15")
    ax.axhline(0.5, ls="--", color="0.55", lw=1.0)
    ax.axhline(0.0, ls=":", color="0.35", lw=1.0)
    s_op = float(np.interp(np.log(R_op), np.log(R), s))
    ax.plot(R_op / 1e3, s_op, "o", color="k", ms=5.5, zorder=5)
    ax.annotate(f"$s$ = {s_op:.2f} at the\noperating point",
                xy=(R_op / 1e3, s_op), xytext=(0.30, 0.22),
                textcoords="axes fraction", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"))
    ax.text(0.97, 0.90, "linear", transform=ax.transAxes, ha="right", fontsize=7,
            color="0.35")
    ax.set_xlabel("force-development rate  $R$  [kN s$^{-1}$]")
    ax.set_ylabel("local exponent  $s = d\\ln v_{\\max}/d\\ln R$")
    ax.set_title("scaling exponent", loc="left")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)

    fig.tight_layout(w_pad=1.9)
    figstyle.panel_letters(fig, axes)
    figstyle.save(fig, "rfd_maxspeed_scaling")
    plt.close(fig)

    print("較正パラメータで Fig. 1 を再生成:")
    print(f"  L_c = {p.L_c:.3f} m, t_a = {p.t_a:.3f} s, beta = {p.beta:.4f}")
    print(f"  F_max = {meas.F_ceiling_Wb:.2f} BW = {p.F_max:.0f} N "
          f"(旧図の既定値は 6000 N = 7.65 BW)")
    print(f"  R_op = {R_op/1e3:.1f} kN/s -> v_op = {v_op:.3f} m/s "
          f"[{regime_op}]  (旧図は R=86 kN/s をハードコード)")
    print(f"  作動点の局所指数 s = {s_op:.3f}")
    print("  output/ と paper/figures/ に rfd_maxspeed_scaling.{png,pdf} を保存。")


if __name__ == "__main__":
    make_figure()
