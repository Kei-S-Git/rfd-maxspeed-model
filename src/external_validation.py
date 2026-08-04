"""
P03 out-of-sample 検証 — 較正に使っていない独立コホートでモデルを検定する。

目的: モデル予測を、較正に使っていない実験データに対して検定する。

`src/degeneracy_audit.py` [D5] が示したとおり、較正後のモデルで検定可能な内容は
容量側の傾き A = beta(1-beta) R [BW/s] の（被験者間・コホート間の）不変性である。
較正点では A = F_avg/T_c に固定される。

本モジュールは公表済みの表から独立コホートを数値化し、次の3つを検定する。

  T1  構造仮定 t_c = L_c/v （接地長 L_c が速度によらずほぼ一定）
  T2  構造仮定 t_a ≈ const （滞空時間が速度によらない）
  T3  普遍 A 仮説: A を Weyand 2010 較正値に凍結して v_max を予測できるか

重要な但し書き（データの性質）:
  * Weyand et al. (2010) の F_avg は計測用トレッドミルで直接測定された値。
  * Paradisis et al. (2019) の FOG（接地平均鉛直力）は式(8) FOG = ST/CT により
    タイミングから計算された量であり、独立した力の測定ではない
    （同論文 Methods, Weyand et al. 2000 に準拠）。したがって
    「力積収支 F_avg = mg(1 + t_a/t_c) が成り立つ」ことを Paradisis で確認しても
    それは恒等式の再確認にすぎず、検証にならない。本モジュールは FOG を
    検証量として用いない。
  * 各コホートの接地長 L_c は L_c = v * t_c として導かれる量である。ただし
    v と t_c は独立に測定されているため、「L_c が能力群間で一定である」ことは
    非自明な経験的事実であり、仮定 t_c = L_c/v の out-of-sample 検定になる。

出典（すべてローカル所蔵 PDF、表から転記）:
  [W2010]  Weyand, Sandell, Prime & Bundle (2010) J Appl Physiol, Table 1
           前方走・最高速度。較正に使用。
  [P2019]  Paradisis, Bissas, Pappas, Zacharogiannis, Theodorou & Girard (2019)
           J Sports Sci, Table 1 / Table 3。成人50名を能力別3群に層別。
           reference/library/paradisis_..._sprint_mechanical_differences_at_maximal_running.pdf
  [K2002]  Kuitunen, Komi & Kyrolainen (2002) Med Sci Sports Exerc.
           男子スプリンター10名 (100m 10.91±0.39 s)。70/80/90/最大の4速度。
           reference/library/kuitunen_komi_inen_2002_knee_and_ankle_joint_stiffness_in.pdf

実行: python src/external_validation.py
"""

from __future__ import annotations

import math
import os
import sys

import figstyle
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class Cohort:
    """最高速度局面のコホート平均値。力は体重比 [Wb]、時間は [s]。"""
    key: str
    source: str
    n: int | None
    v_max: float          # 最高疾走速度 [m/s]   (測定)
    t_c: float            # 接地時間 [s]        (測定)
    t_a: float            # 滞空時間 [s]        (測定)
    L_c_reported: float | None = None   # 論文が報告する接地長 [m]
    F_avg_measured: float | None = None  # 直接測定された接地平均鉛直力 [Wb]
    note: str = ""
    calibration: bool = False

    @property
    def L_c(self) -> float:
        """接地長 [m]。報告値があればそれを、無ければ v*t_c を用いる。"""
        return self.L_c_reported if self.L_c_reported is not None else self.v_max * self.t_c

    @property
    def A_required(self) -> float:
        """
        力積収支が要求する容量側の傾き A = F_avg/t_c = (t_c + t_a)/t_c^2 [Wb/s]。
        mg = 1 Wb。タイミングのみから決まる。
        """
        return (self.t_c + self.t_a) / self.t_c**2

    @property
    def F_avg_identity(self) -> float:
        """定常走行の恒等式 F_avg = mg(1 + t_a/t_c) [Wb]。"""
        return 1.0 + self.t_a / self.t_c


# ---------------------------------------------------------------------------
# データ（公表表からの転記）
# ---------------------------------------------------------------------------
COHORTS: list[Cohort] = [
    Cohort(
        key="W2010 (calibration)", source="Weyand et al. 2010 Table 1", n=None,
        v_max=9.20, t_c=0.108, t_a=0.119, L_c_reported=0.98,
        F_avg_measured=2.08, calibration=True,
        note="計測用トレッドミル。F_avg は直接測定。本モデルの較正元。",
    ),
    Cohort(
        key="P2019 Slow", source="Paradisis et al. 2019 Table 1", n=13,
        v_max=7.67, t_c=0.134, t_a=0.127, L_c_reported=1.03,
        note="地上走。FOG=1.952 Wb は ST/CT による計算値のため検証に使わない。",
    ),
    Cohort(
        key="P2019 Medium", source="Paradisis et al. 2019 Table 1", n=21,
        v_max=8.44, t_c=0.124, t_a=0.118, L_c_reported=1.05,
        note="地上走。",
    ),
    Cohort(
        key="P2019 Fast", source="Paradisis et al. 2019 Table 1", n=16,
        v_max=9.37, t_c=0.107, t_a=0.124, L_c_reported=1.00,
        note="地上走。",
    ),
    Cohort(
        key="K2002 max", source="Kuitunen et al. 2002 Results", n=10,
        v_max=9.73, t_c=0.094, t_a=0.129, L_c_reported=None,
        note="地上走。最大速度時。L_c は v*t_c として導出。",
    ),
]

# Kuitunen の被験者内・速度系列（70% と 100% のみ本文に数値あり）
KUITUNEN_SERIES = [
    # (速度 [m/s], t_c [s], t_a [s], 説明)
    (7.00, 0.131, 0.172, "70% of max"),
    (9.73, 0.094, 0.129, "max"),
]

# 較正値（src/calibrate_parameters.py と一致）
A_CALIBRATED = 2.08 / 0.108  # = F_avg/T_c [Wb/s]


def predict_v_max(t_a: float, L_c: float, A: float = A_CALIBRATED) -> float:
    """
    A を凍結し、コホート固有の (t_a, L_c) から v_max を予測する。
    A t_c^2 = mg (t_c + t_a),  mg = 1 Wb  ->  A t_c^2 - t_c - t_a = 0
    """
    disc = 1.0 + 4.0 * A * t_a
    t_c = (1.0 + math.sqrt(disc)) / (2.0 * A)
    return L_c / t_c


def _hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def t1_contact_length() -> None:
    _hdr("[T1] 構造仮定 t_c = L_c/v : 接地長は能力群間で一定か")
    print(f"  {'cohort':<20} {'n':>4} {'v_max':>7} {'t_c[ms]':>8} "
          f"{'L_c rep':>8} {'v*t_c':>8} {'diff':>7}")
    for c in COHORTS:
        vt = c.v_max * c.t_c
        rep = f"{c.L_c_reported:.3f}" if c.L_c_reported is not None else "   -  "
        dif = (f"{(c.L_c_reported - vt)/vt*100:+.1f}%"
               if c.L_c_reported is not None else "   -  ")
        print(f"  {c.key:<20} {c.n if c.n else '-':>4} {c.v_max:>7.2f} "
              f"{c.t_c*1e3:>8.0f} {rep:>8} {vt:>8.3f} {dif:>7}")

    ext = [c for c in COHORTS if not c.calibration]
    lcs = [c.L_c for c in ext]
    print(f"\n  独立コホートの L_c: {', '.join(f'{x:.3f}' for x in lcs)} m")
    print(f"  平均 {sum(lcs)/len(lcs):.3f} m, レンジ {max(lcs)-min(lcs):.3f} m "
          f"({(max(lcs)-min(lcs))/(sum(lcs)/len(lcs))*100:.1f}% of mean)")
    v_span = (max(c.v_max for c in ext) - min(c.v_max for c in ext))
    print(f"  この間に v_max は {v_span:.2f} m/s "
          f"({v_span/min(c.v_max for c in ext)*100:.0f}%) 変化している。")
    print("  Paradisis 2019 は接地長の群間差を有意でないと報告 (Table 1)。")
    print("  => 仮定 t_c = L_c/v は独立コホートで支持される。")


def t2_aerial_time() -> None:
    _hdr("[T2] 構造仮定 t_a ≈ const : 滞空時間は速度によらないか")
    ext = [c for c in COHORTS if not c.calibration]
    print("  (a) 能力群間（各自の最高速度での比較）")
    for c in COHORTS:
        print(f"      {c.key:<20} v={c.v_max:>5.2f} m/s   t_a={c.t_a*1e3:>5.0f} ms")
    tas = [c.t_a for c in ext]
    print(f"      独立コホートのレンジ {min(tas)*1e3:.0f}-{max(tas)*1e3:.0f} ms "
          f"({(max(tas)-min(tas))/(sum(tas)/len(tas))*100:.1f}% of mean)")
    print("      Paradisis 2019 は滞空時間の群間差を有意でないと報告 (Table 1)。")
    print("      => 最高速度どうしの比較では支持される。")

    print("\n  (b) 被験者内・速度系列 [K2002]")
    for v, tc, ta, lab in KUITUNEN_SERIES:
        print(f"      {lab:<12} v={v:>5.2f} m/s  t_c={tc*1e3:>5.0f} ms  "
              f"t_a={ta*1e3:>5.0f} ms  L_c={v*tc:.3f} m")
    v0, tc0, ta0, _ = KUITUNEN_SERIES[0]
    v1, tc1, ta1, _ = KUITUNEN_SERIES[-1]
    print(f"      t_a は {ta0*1e3:.0f} -> {ta1*1e3:.0f} ms ({(ta1/ta0-1)*100:+.0f}%, "
          f"Kuitunen は P<0.001 と報告) => 被験者内では一定でない。")
    print(f"      一方 L_c は {v0*tc0:.3f} -> {v1*tc1:.3f} m でほぼ不変。")
    print("      => t_a 一定は「最高速度どうしの比較」に限って成り立つ近似であり、")
    print("         同一被験者の亜最大速度域には適用できない。原稿はこれを区別していない。")


def t3_universal_A() -> None:
    _hdr("[T3] 普遍 A 仮説の検定 : A を較正値に凍結して v_max を予測できるか")
    print(f"  較正値 A = F_avg/T_c = {A_CALIBRATED:.4f} Wb/s  [W2010]\n")
    print(f"  {'cohort':<20} {'v meas':>8} {'v pred':>8} {'error':>8} "
          f"{'A required':>11} {'A/A_cal':>8}")
    errs = []
    for c in COHORTS:
        vp = predict_v_max(c.t_a, c.L_c)
        err = (vp - c.v_max) / c.v_max * 100
        if not c.calibration:
            errs.append(err)
        print(f"  {c.key:<20} {c.v_max:>8.2f} {vp:>8.2f} {err:>+7.1f}% "
              f"{c.A_required:>11.2f} {c.A_required/A_CALIBRATED:>8.3f}")

    rmse = math.sqrt(sum(e**2 for e in errs) / len(errs))
    bias = sum(errs) / len(errs)
    print(f"\n  独立コホートのみ: bias {bias:+.1f}%, RMSE {rmse:.1f}%, "
          f"レンジ {min(errs):+.1f}% ... {max(errs):+.1f}%")

    ext = [c for c in COHORTS if not c.calibration]
    A_lo = min(c.A_required for c in ext)
    A_hi = max(c.A_required for c in ext)
    print(f"\n  必要 A は {A_lo:.2f} - {A_hi:.2f} Wb/s の範囲で変化 "
          f"({(A_hi/A_lo-1)*100:.0f}% の幅)。")
    print("  => A は被験者間で不変ではない。凍結 A のモデルは遅い群を大きく過大予測する。")
    print("     これは否定的結果として報告すべきであり、同時に『速度差は容量 A の差で")
    print("     説明される』という本モデルの主張と整合する。ただし A は同じ")
    print("     タイミングから導かれるため、これ自体は独立な確証ではない。")

    print("\n  A と v_max の関係:")
    for c in sorted(COHORTS, key=lambda x: x.v_max):
        print(f"      v={c.v_max:>5.2f} m/s -> A={c.A_required:>6.2f} Wb/s"
              f"{'   (calibration)' if c.calibration else ''}")


def t4_scaling_of_required_capacity() -> None:
    """必要容量 A の速度スケーリング（独立コホート横断）。"""
    _hdr("[T4] 必要容量 A の速度スケーリング（falsifiable prediction）")
    pts = sorted(((c.v_max, c.A_required, c.key) for c in COHORTS))
    n = len(pts)
    lx = [math.log(v) for v, _, _ in pts]
    ly = [math.log(a) for _, a, _ in pts]
    mx, my = sum(lx) / n, sum(ly) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    slope = sxy / sxx
    icpt = my - slope * mx
    syy = sum((b - my) ** 2 for b in ly)
    r2 = sxy**2 / (sxx * syy)
    print(f"  5コホート (4研究室) 横断の log-log 回帰:")
    print(f"      ln A = {slope:.3f} ln v + {icpt:.3f}     R^2 = {r2:.4f}")
    print(f"      => A ∝ v^{slope:.2f}")
    print("\n  モデルの解析形 A = v/L_c + t_a v^2/L_c^2 は、L_c, t_a 一定のとき")
    print("  v^1 と v^2 の中間の指数を与える。観測値はこの帯に収まっている。")
    print("\n  反証可能な予測: 全身鉛直力の立ち上がり率（体重比）は最高速度に対し")
    print(f"  およそ v^{slope:.1f} で増加しなければならない。7.67 -> 9.73 m/s の間で")
    print(f"  {(pts[-1][1]/pts[0][1]-1)*100:.0f}% の増加を要する。これは同一被験者で")
    print("  全身鉛直GRF立ち上がり率と最高速度を測れば直接反証できる。")


def summary() -> None:
    ext = [c for c in COHORTS if not c.calibration]
    lcs = [c.L_c for c in ext]
    vs = [c.v_max for c in ext]
    As = [c.A_required for c in COHORTS]
    _hdr("結論")
    print(f"  T1 接地長一定  : 独立コホートで支持"
          f"（L_c {min(lcs):.3f}-{max(lcs):.3f} m、その間 v_max は "
          f"{(max(vs)-min(vs))/min(vs)*100:.0f}% 変化）")
    print("  T2 滞空時間一定: 最高速度どうしの比較では支持。被験者内の亜最大速度域では不成立")
    print(f"  T3 普遍 A      : 棄却。A は {min(As):.1f}-{max(As):.1f} Wb/s で変化し、"
          f"凍結 A は遅い群を過大予測")
    print("  T4 スケーリング: 必要 A は v のべき乗で増加（反証可能な予測として提示可能）")
    print()
    print("  => モデルの構造仮定は独立データで支持されるが、v_max を予測するには")
    print("     コホート固有の A が必要であり、A は本データからは独立に測れない。")
    print("     報告としては『構造仮定は out-of-sample で検証済み、")
    print("     ただし中心量 A の独立測定は未達』と正直に報告するのが妥当。")
    print()
    print("  未達成の検証（新規データ取得が必要）:")
    print("    全身鉛直GRF立ち上がり率と最高速度を同一被験者で測定したデータセットは")
    print("    ローカル所蔵に存在しない。ライブラリの RFD 論文は等尺性・跳躍 RFD")
    print("    （別構成概念）である。")


def _power_fit() -> tuple[float, float, float]:
    """log-log 回帰 ln A = s ln v + c を返す (slope, intercept, R^2)。"""
    pts = [(c.v_max, c.A_required) for c in COHORTS]
    n = len(pts)
    lx = [math.log(v) for v, _ in pts]
    ly = [math.log(a) for _, a in pts]
    mx, my = sum(lx) / n, sum(ly) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    syy = sum((b - my) ** 2 for b in ly)
    slope = sxy / sxx
    return slope, my - slope * mx, sxy**2 / (sxx * syy)


def make_figure() -> None:
    """
    Out-of-sample 検証図（3パネル）。

    図版監査で指摘された既存図の欠陥を回避する:
      - 単位は BW（体重比）。Wb は weber の SI 記号であり誤用
      - 最終掲載寸法で本文フォント 8 pt 相当を確保（figsize を段幅に合わせる）
      - グレースケール印刷でも判別できるようマーカー形状と線種で区別
      - ベクタ (PDF) とラスタ (600 dpi PNG) の両方を出力
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figstyle.apply()

    ext = [c for c in COHORTS if not c.calibration]
    cal = [c for c in COHORTS if c.calibration][0]

    fig, axes = plt.subplots(1, 3, figsize=figstyle.figsize(178, 63))

    # Panel letters are placed in figure coordinates after tight_layout so they
    # cannot collide with the y-axis labels (JEB: 12 pt bold uppercase).
    _letters = []

    def panel(ax, letter):
        _letters.append((ax, letter))

    # ---- (a) 構造仮定: L_c と t_a は最高速度によらないか -------------------
    ax = axes[0]
    ax.plot([c.v_max for c in ext], [c.L_c for c in ext], "o",
            ms=4.5, mfc="0.25", mec="0.25", label="$L_c$ (independent)")
    ax.plot([cal.v_max], [cal.L_c], "o", ms=5.5, mfc="none", mec="0.25",
            mew=1.1, label="$L_c$ (calibration)")
    lcs = [c.L_c for c in COHORTS]
    ax.axhline(sum(lcs) / len(lcs), color="0.25", ls="--", lw=0.8)
    ax.set_ylim(0.85, 1.15)
    ax.set_xlabel("measured top speed  [m s$^{-1}$]")
    ax.set_ylabel("contact length  $L_c$  [m]", color="0.25")
    ax2 = ax.twinx()
    ax2.plot([c.v_max for c in ext], [c.t_a * 1e3 for c in ext], "^",
             ms=4.5, mfc="0.65", mec="0.35", label="$t_a$")
    ax2.plot([cal.v_max], [cal.t_a * 1e3], "^", ms=5.5, mfc="none",
             mec="0.35", mew=1.1, label="$t_a$, calib.")
    tas = [c.t_a * 1e3 for c in COHORTS]
    ax2.axhline(sum(tas) / len(tas), color="0.55", ls=":", lw=0.9)
    ax2.set_ylim(100, 170)
    ax2.set_ylabel("aerial time  $t_a$  [ms]", color="0.45")
    ax2.tick_params(labelsize=7.5)
    ax.set_title("structural assumptions", loc="left")
    panel(ax, "A")
    ax.grid(True, alpha=0.25, lw=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, ["$L_c$", "$L_c$, calib.", "$t_a$", "$t_a$, calib."],
              loc="lower center", ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=0.9, borderpad=0.1)

    # ---- (b) 凍結 A による予測 vs 実測 (1:1) --------------------------------
    ax = axes[1]
    lo, hi = 7.0, 10.2
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.9, label="identity")
    ax.plot([lo, hi], [lo * 1.1, hi * 1.1], "k:", lw=0.7, label="$\\pm10\\%$")
    ax.plot([lo, hi], [lo * 0.9, hi * 0.9], "k:", lw=0.7)
    for c in ext:
        ax.plot(c.v_max, predict_v_max(c.t_a, c.L_c), "s",
                ms=4.5, mfc="0.25", mec="0.25")
    ax.plot(cal.v_max, predict_v_max(cal.t_a, cal.L_c), "s",
            ms=5.5, mfc="none", mec="0.25", mew=1.1)
    errs = [(predict_v_max(c.t_a, c.L_c) - c.v_max) / c.v_max * 100 for c in ext]
    rmse = math.sqrt(sum(e**2 for e in errs) / len(errs))
    ax.text(0.03, 0.975,
            f"bias {sum(errs)/len(errs):+.1f}%,  RMSE {rmse:.1f}%",
            transform=ax.transAxes, va="top", fontsize=6.8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("measured top speed  [m s$^{-1}$]")
    ax.set_ylabel("predicted, capacity frozen  [m s$^{-1}$]")
    ax.set_title("universal-capacity test", loc="left")
    panel(ax, "B")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)

    # ---- (c) 必要容量のスケーリング ----------------------------------------
    ax = axes[2]
    vv = np.linspace(7.2, 10.2, 80)
    band_lo = vv / 1.05 + 0.118 * vv**2 / 1.05**2
    band_hi = vv / 0.92 + 0.129 * vv**2 / 0.92**2
    ax.fill_between(vv, band_lo, band_hi, color="0.80", alpha=0.7,
                    label="model, $L_c$/$t_a$ range")
    s, c0, r2 = _power_fit()
    ax.plot(vv, np.exp(c0) * vv**s, "k--", lw=1.0,
            label=f"fit  $A\\propto v^{{{s:.2f}}}$")
    ax.plot([c.v_max for c in ext], [c.A_required for c in ext], "D",
            ms=4.2, mfc="0.25", mec="0.25")
    ax.plot([cal.v_max], [cal.A_required], "D", ms=5.2, mfc="none",
            mec="0.25", mew=1.1)
    ax.text(0.04, 0.94, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
            va="top", fontsize=7)
    ax.set_xlabel("measured top speed  [m s$^{-1}$]")
    ax.set_ylabel("required capacity  $A$  [BW s$^{-1}$]")
    ax.set_title("required capacity", loc="left")
    panel(ax, "C")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.87), frameon=False,
              handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    ax.grid(True, alpha=0.25, lw=0.5)

    fig.tight_layout(w_pad=1.9)
    figstyle.panel_letters(fig, [a for a, _ in _letters],
                           [l for _, l in _letters])

    figstyle.save(fig, "external_validation")
    plt.close(fig)
    print("\n図を output/ と paper/figures/ に external_validation.{png,pdf} として保存。")


def main() -> None:
    t1_contact_length()
    t2_aerial_time()
    t3_universal_A()
    t4_scaling_of_required_capacity()
    summary()
    make_figure()


if __name__ == "__main__":
    main()
