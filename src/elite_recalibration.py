"""
世界エリートへの再較正：RFD律速力積モデルを最高速度 6-11.5 m/s 域で検証する。

[W2010] の単一整合点（9.20 m/s）から、[W2000] の力-速度回帰（33名、6.2-11.1 m/s）へ
較正を拡張し、世界エリート（~11.5 m/s）への外挿と必要RFDの予測を行う。

一次資料（全文共有ライブラリ `reference/library/` 収録）:
  [W2000] Weyand et al. (2000) J Appl Physiol 89:1991. トレッドミル33名, 6.2-11.1 m/s。
          Favg/Wb = 1.26 + 0.101 v  (v=最高速度[m/s], R^2=0.39)。
          平均 aerial time = 0.128 s（速度に非依存）。
          Level 0°: Lc=0.99 m, T_c=0.107 s, Favg=2.14 Wb @ 9.25 m/s。
  [W2010] Weyand et al. (2010). 形状因子 kappa=F_peak/F_avg=1.74 (beta=0.425),
          力の天井 F_ceiling=4.20 Wb（片脚ホッピングピーク）。
  [Salo2010] Salo et al. (2010) 世界トップ100m選手 52名。v=SF×SL、エリートはSL依存。
  [Kor2009] Korhonen et al. (2009). 最高走速度は地面反力低下・接地時間延長と関連する
          一方、等尺性RFDは最高速度と無相関（最大等尺力のみ相関）と報告。→律速は全身の
          地面反力立ち上がり率であり単関節等尺性RFDではない、との本モデルの区別を支持。
"""

from __future__ import annotations

import numpy as np

import figstyle
from rfd_maxspeed_model import SprintParams


# ---- [W2000] 実測回帰・定数 -------------------------------------------------
def favg_weyand2000(v: float) -> float:
    """[W2000] 経験回帰: 接地平均鉛直力 F_avg/Wb = 1.26 + 0.101 v。"""
    return 1.26 + 0.101 * v


TA_ELITE = 0.128      # aerial time [s] ([W2000], 速度非依存)
KAPPA = 1.74          # 波形形状因子 ([W2010])
BETA = 1.0 - 1.0 / KAPPA   # = 0.425
F_CEILING_WB = 4.20   # 力の天井 [Wb] ([W2010] ホッピングピーク)


def required_favg_model(v: float, L_c: float, t_a: float = TA_ELITE) -> float:
    """モデル(力積収支)による必要 F_avg/Wb = 1 + t_a v / L_c。"""
    t_c = L_c / v
    return 1.0 + t_a / t_c  # = 1 + t_a v / L_c


def required_R_Wb_per_s(v: float, L_c: float, t_a: float = TA_ELITE,
                        kappa: float = KAPPA, beta: float = BETA) -> float:
    """
    最高速度 v を実現するのに必要な立ち上がり率 R [Wb/s]（質量非依存）。
    必要 F_peak = kappa * F_avg = kappa * (1 + t_a v/L_c)。
    rate-limited: F_peak = R * beta * t_c  →  R = F_peak / (beta * t_c)。
    """
    t_c = L_c / v
    F_peak_Wb = kappa * required_favg_model(v, L_c, t_a)
    return F_peak_Wb / (beta * t_c)


def required_Fpeak_Wb(v: float, L_c: float, t_a: float = TA_ELITE,
                      kappa: float = KAPPA) -> float:
    return kappa * required_favg_model(v, L_c, t_a)


def main() -> None:
    print("=" * 74)
    print("世界エリートへの再較正 (Weyand 2000 回帰 + Weyand 2010 形状/天井)")
    print("=" * 74)

    L_c = 1.0   # エリート代表接地長 [m] ([W2000] level 0.99, 高速でわずかに増加)
    print(f"\n較正定数: L_c={L_c} m, t_a={TA_ELITE} s, kappa={KAPPA}, "
          f"beta={BETA:.3f}, F_ceiling={F_CEILING_WB} Wb")

    # --- 検証A: モデルの F_avg(v) を [W2000] 経験回帰と照合 (6-11 m/s) ---
    print("\n[検証A] 必要 F_avg/Wb : モデル(力積収支) vs Weyand2000 回帰")
    print(f"{'v [m/s]':>8} {'model':>8} {'W2000':>8} {'diff':>7}")
    for v in [7.0, 8.0, 9.0, 9.25, 10.0, 11.0, 11.1]:
        m = required_favg_model(v, L_c)
        w = favg_weyand2000(v)
        print(f"{v:>8.2f} {m:>8.2f} {w:>8.2f} {(m-w):>+7.2f}")
    # 測定域(8-11 m/s)平均差
    vv = np.linspace(8, 11, 100)
    diff = np.array([required_favg_model(v, L_c) - favg_weyand2000(v) for v in vv])
    print(f"  測定域 8-11 m/s の平均|差| = {np.mean(np.abs(diff)):.3f} Wb "
          f"(回帰の R^2=0.39 の散らばり内)")

    # --- 検証B: 必要RFD R(v) は最高速度とともに増大するか ---
    print("\n[検証B] 最高速度に必要な立ち上がり率 R(v) [Wb/s]")
    print(f"{'v [m/s]':>8} {'R req':>9} {'F_peak req':>11} {'vs ceiling':>12}")
    ref = {}
    for tag, v in [("recreational", 9.20), ("Weyand2000 fastest", 11.1),
                   ("world elite", 11.5), ("world record ~", 12.0)]:
        R = required_R_Wb_per_s(v, L_c)
        Fp = required_Fpeak_Wb(v, L_c)
        flag = "OK (<ceiling)" if Fp < F_CEILING_WB else "≥ceiling!"
        ref[v] = R
        print(f"{v:>8.2f} {R:>9.1f} {Fp:>10.2f}W {flag:>13}  [{tag}]")
    print(f"  → エリート11.5 vs レク9.2: 必要RFDは "
          f"{ref[11.5]/ref[9.20]:.2f}倍 ({(ref[11.5]/ref[9.20]-1)*100:.0f}%増)")

    # --- Lc 感度 (エリートで支配的な不確かさ) ---
    print("\n[Lc 感度] 世界エリート 11.5 m/s での必要RFD・必要ピーク力")
    for L in [0.95, 1.00, 1.05, 1.10]:
        R = required_R_Wb_per_s(11.5, L)
        Fp = required_Fpeak_Wb(11.5, L)
        print(f"  L_c={L:.2f} m : R={R:6.1f} Wb/s, F_peak={Fp:.2f} Wb "
              f"({'<' if Fp<F_CEILING_WB else '≥'} 天井 {F_CEILING_WB})")

    print("\n[結論]")
    print("  ・モデルの力積収支は Weyand2000 の力-速度回帰を測定域全体で再現。")
    print("  ・最高速度に必要な RFD は速度とともに単調増大し、エリートはレク走者の")
    print("    約1.3-1.5倍のRFDを要する。注: Korhonen2009では等尺性RFDは最高速度と")
    print("    無相関（最大等尺力は相関）——律速するのは全身の地面反力立ち上がり率で")
    print("    あって単関節等尺性RFDではない、という本モデルの区別と整合。")
    print("  ・エリート域では必要ピーク力が天井(4.20 Wb)に接近 → 高RFDに加えて")
    print("    より高い筋力天井も要する（強度とRFDの共律速）。ただし十分な筋力があれば")
    print("    律速はRFD側（rate-limited）。")

    _make_figure(L_c)
    print("\n図を output/elite_recalibration.png に保存しました。")


def _make_figure(L_c: float) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    figstyle.apply()
    v = np.linspace(6.0, 12.0, 200)
    fig, axes = plt.subplots(1, 3, figsize=figstyle.figsize(178, 63))

    # (a) F_avg(v): モデル vs Weyand2000 回帰
    ax = axes[0]
    ax.plot(v, [required_favg_model(x, L_c) for x in v], "-", color="0.15",
            label="model")
    ax.plot(v, [favg_weyand2000(x) for x in v], "--", color="0.45",
            label="Weyand 2000 ($R^2$ = 0.39)")
    ax.set_xlabel("top speed  $v$  [m s$^{-1}$]")
    ax.set_ylabel("required $F_{\\mathrm{avg}}$  [BW]")
    ax.set_title("impulse balance vs regression", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.5)

    # (b) 必要RFD R(v)
    ax = axes[1]
    ax.plot(v, [required_R_Wb_per_s(x, L_c) for x in v], "-", color="0.15")
    for vv, lab in [(9.20, "rec. 9.2"), (11.5, "elite 11.5")]:
        R = required_R_Wb_per_s(vv, L_c)
        ax.plot(vv, R, "o", ms=5, color="k", zorder=5)
        ax.annotate(f"{lab}\n{R:.0f} BW/s", (vv, R),
                    textcoords="offset points", xytext=(-40, -16), fontsize=7)
    ax.set_xlabel("top speed  $v$  [m s$^{-1}$]")
    ax.set_ylabel("required $R$  [BW s$^{-1}$]")
    ax.set_title("required rate (algebraic consequence)", loc="left")
    ax.grid(True, alpha=0.25, lw=0.5)

    # (c) 必要ピーク力 vs 天井
    ax = axes[2]
    ax.plot(v, [required_Fpeak_Wb(x, L_c) for x in v], "-", color="0.15",
            label="required")
    ax.axhline(F_CEILING_WB, ls="--", color="0.45",
               label="ceiling (hopping)")
    ax.set_xlabel("top speed  $v$  [m s$^{-1}$]")
    ax.set_ylabel("required $F_{\\mathrm{peak}}$  [BW]")
    ax.set_title("required peak force vs ceiling", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.5)

    fig.tight_layout(w_pad=1.9)
    figstyle.panel_letters(fig, axes)
    figstyle.save(fig, "elite_recalibration")
    plt.close(fig)


if __name__ == "__main__":
    main()
