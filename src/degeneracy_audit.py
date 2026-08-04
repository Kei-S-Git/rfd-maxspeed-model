"""
P03 縮退監査 (degeneracy audit).

投稿原稿 paper/paper_en.tex が「3つの独立な検証 (three independent checks)」と
呼ぶもののうち何が実質的な検証で、何が較正手続きから恒等的に従うだけかを、
sympy による記号計算で切り分ける。

背景:
  独立データによる検証を設計する前に、既存の「検証」と称する量が実際に何を
  測っているかを確定する必要がある。較正手続きから恒等的に従う量を検証と
  呼んでしまうと、後段の out-of-sample 検定の設計自体を誤るためである。

主要な結論 (すべて下記で機械検証される):

  [D1] 較正 kappa = F_peak/F_avg, beta = 1 - 1/kappa, R = F_peak/(beta*T_c)
       を rate-limited 分岐の到達可能力積 J_avail = beta(1-beta) R t_c^2 に
       代入すると、beta と R が相殺し

            J_avail = F_avg * t_c^2 / T_c

       となる。すなわち「較正された RFD 律速モデル」の力積収支は
       R にも beta にも依存しない。

  [D2] ゆえに検証2 (較正モデルが実測最高速度 9.20 m/s を 2% で再現する) は
       検証1 (力積収支の予測 F_avg が実測 F_avg と 1.8% で一致する) と
       同一の言明であり、RFD の内容を一切含まない。両者が報告する残差は
       いずれも「Weyand 2010 Table 1 内部で F_avg^meas と定常走行の恒等式
       mg(1 + t_a/t_c) がどれだけずれているか」である。

  [D3] 非循環性の論証 —— 必要レート R_req = F_peak/(beta*T_c) と、波形から
       「直接読んだ」レート R_wave = F_peak/t_rise^meas の比 —— は

            R_req / R_wave = t_rise^meas / (beta * T_c)

       であり、共通因子 F_peak が消える。t_rise^meas = T_c/2 を採るとこれは
       kappa / (2(kappa-1)) となり、"20% 以内で一致" は
       「kappa が 2 からどれだけ離れているか」の言い換えにすぎない。
       生理学的な立ち上がり率の実在性を何も検定していない。

  [D4] 「エリート速度では必要 RFD が 30-40% 高い」は、kappa, beta, L_c, t_a を
       速度非依存に固定した下での R(v) = kappa (1 + t_a v / L_c) v / (beta L_c)
       の数値評価であり、v の 2 次式として構成上単調増加する。結果ではなく代数。

  [D5] 縮退していない (＝真に検証可能な) 内容は何か:
       rate 律速の実質は「到達可能な平均力が接地時間に比例する」

            F_avail(t_c) = beta(1-beta) R t_c            (容量側)

       と、力積収支が要求する

            F_req(t_c) = mg (1 + t_a / t_c)               (要求側)

       の交点として v_max が決まる、という構造にある。容量側の傾き
       A = beta(1-beta) R [Wb/s] は較正点で A = F_avg^meas / T_c に固定されるが、
       t_a と L_c が異なる別コホートに対して A を凍結したまま v_max を予測すれば、
       それは out-of-sample の予測になる。これが検証の設計である。

実行: python src/degeneracy_audit.py
"""

from __future__ import annotations

import math
import sys

import sympy as sp

# Windows の既定コンソール (cp932) では日本語・数学記号が出力できないため。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Weyand et al. (2010) Table 1, forward running at top speed.
# calibrate_parameters.Measured と同一の値（体重比 Wb）。
# ---------------------------------------------------------------------------
MEAS = dict(
    F_avg=2.08,      # 接地平均鉛直力 [Wb]
    F_peak=3.62,     # 鉛直力ピーク [Wb]
    F_ceiling=4.20,  # 片脚ホッピングのピーク力 [Wb]
    L_c=0.98,        # 接地長 [m]
    T_c=0.108,       # 接地時間 [s]
    T_aer=0.119,     # 滞空時間 [s]
    v_top=9.20,      # 最高速度 [m/s]
)
G = 9.81
# 原稿 L288 が「波形から直接読んだ」とする立ち上がり時間 [s]（本文の ~55 ms）。
T_RISE_MEAS = 0.055


def _hdr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# [D1] beta と R の相殺（記号）
# ---------------------------------------------------------------------------
def d1_symbolic_cancellation() -> sp.Expr:
    _hdr("[D1] 較正を代入すると beta と R が相殺する（記号計算）")

    F_avg, F_peak, T_c, t_c = sp.symbols("F_avg F_peak T_c t_c", positive=True)

    kappa = F_peak / F_avg
    beta = 1 - 1 / kappa
    R = F_peak / (beta * T_c)

    # rfd_maxspeed_model.available_impulse の rate-limited 分岐
    J_avail = beta * (1 - beta) * R * t_c**2

    reduced = sp.simplify(J_avail)
    target = F_avg * t_c**2 / T_c
    diff = sp.simplify(reduced - target)

    print(f"  kappa    = {kappa}")
    print(f"  beta     = {sp.simplify(beta)}")
    print(f"  R        = {sp.simplify(R)}")
    print(f"  J_avail  = beta(1-beta) R t_c^2")
    print(f"           = {reduced}")
    print(f"  target   = F_avg t_c^2 / T_c = {target}")
    print(f"  simplify(J_avail - target) = {diff}")
    assert diff == 0, "beta/R の相殺が成立しない"
    print("  => 恒等的に一致。J_avail は R にも beta にも依存しない。 [OK]")

    # 係数 A = beta(1-beta) R が F_avg/T_c に等しいことも明示
    A = sp.simplify(beta * (1 - beta) * R)
    assert sp.simplify(A - F_avg / T_c) == 0
    print(f"  容量側の傾き A = beta(1-beta) R = {A}  (= F_avg/T_c)")
    return A


# ---------------------------------------------------------------------------
# [D2] 検証2 は検証1 の言い換え（記号 + 数値）
# ---------------------------------------------------------------------------
def d2_check2_is_check1() -> None:
    _hdr("[D2] 検証2（最高速度の再現）は検証1（力積収支）と同一の言明")

    F_avg, T_c, t_a, L_c, t_c, v = sp.symbols(
        "F_avg T_c t_a L_c t_c v", positive=True)

    # 力積収支: J_avail(t_c) = mg (t_c + t_a)   ただし mg = 1 Wb
    eq = sp.Eq(F_avg * t_c**2 / T_c, t_c + t_a)
    sol = [s for s in sp.solve(eq, t_c) if s.is_real is not False]
    t_c_sol = sp.simplify([s for s in sol if sp.simplify(s.subs(
        {F_avg: 2.08, T_c: 0.108, t_a: 0.119})) > 0][0])
    print(f"  解 t_c = {t_c_sol}")
    print("  → 右辺・左辺とも R, beta, F_peak を含まない。")

    num = {F_avg: MEAS["F_avg"], T_c: MEAS["T_c"], t_a: MEAS["T_aer"]}
    t_c_num = float(t_c_sol.subs(num))
    v_num = MEAS["L_c"] / t_c_num
    print(f"\n  数値: t_c = {t_c_num:.5f} s,  v_max = L_c/t_c = {v_num:.3f} m/s")
    print(f"        原稿の報告値 9.01 m/s と一致: {abs(v_num - 9.01) < 5e-3}")
    assert abs(v_num - 9.01) < 5e-3

    # 検証1 の残差
    t_c_meas = MEAS["L_c"] / MEAS["v_top"]
    F_avg_identity = 1.0 + MEAS["T_aer"] / t_c_meas
    err1 = abs(F_avg_identity - MEAS["F_avg"]) / MEAS["F_avg"] * 100
    err2 = abs(v_num - MEAS["v_top"]) / MEAS["v_top"] * 100
    print(f"\n  検証1 残差: 恒等式 mg(1+t_a/t_c) = {F_avg_identity:.4f} Wb "
          f"vs 実測 F_avg = {MEAS['F_avg']:.2f} Wb  -> {err1:.1f}%")
    print(f"  検証2 残差: 予測 v_max = {v_num:.3f} vs 実測 {MEAS['v_top']:.2f} m/s "
          f"-> {err2:.1f}%")
    print("  両者は同一の内部不整合（Table 1 の F_avg と定常恒等式のずれ）の")
    print("  2通りの表示であり、独立ではない。")


# ---------------------------------------------------------------------------
# [D3] 非循環性の比は立ち上がり時間の2定義の比
# ---------------------------------------------------------------------------
def d3_circularity_ratio() -> None:
    _hdr("[D3] 「20%以内で一致」は kappa が 2 からどれだけ離れているかの言い換え")

    F_peak, F_avg, T_c, t_rise_meas = sp.symbols(
        "F_peak F_avg T_c t_rise_meas", positive=True)
    kappa = F_peak / F_avg
    beta = 1 - 1 / kappa

    R_req = F_peak / (beta * T_c)          # モデルが要求するレート
    R_wave = F_peak / t_rise_meas          # 「波形から直接読んだ」レート
    ratio = sp.simplify(R_req / R_wave)
    print(f"  R_req / R_wave = {ratio}")
    assert sp.simplify(ratio - t_rise_meas / (beta * T_c)) == 0
    print("  → 共通因子 F_peak が相殺し、立ち上がり時間の2定義の比のみが残る。")

    ratio_half = sp.simplify(ratio.subs(t_rise_meas, T_c / 2))
    kap = sp.symbols("kappa", positive=True)
    ratio_kappa = sp.simplify(ratio_half.rewrite(sp.Pow).subs(
        F_peak, kap * F_avg))
    print(f"  t_rise^meas = T_c/2 と採ると  ratio = {sp.simplify(ratio_kappa)}"
          f"  (= kappa / (2(kappa-1)))")

    k = MEAS["F_peak"] / MEAS["F_avg"]
    b = 1 - 1 / k
    r_req = MEAS["F_peak"] / (b * MEAS["T_c"])
    r_half = MEAS["F_peak"] / (MEAS["T_c"] / 2)
    r_prose = MEAS["F_peak"] / T_RISE_MEAS
    print(f"\n  数値: kappa = {k:.4f}, beta = {b:.4f}, beta*T_c = {b*MEAS['T_c']*1e3:.1f} ms")
    print(f"        R_req  = {r_req:.1f} Wb/s")
    print(f"        R_wave (T_c/2 = {MEAS['T_c']/2*1e3:.1f} ms) = {r_half:.1f} Wb/s"
          f"  -> 比 {r_req/r_half:.3f}")
    print(f"        R_wave (本文 {T_RISE_MEAS*1e3:.0f} ms)      = {r_prose:.1f} Wb/s"
          f"  -> 比 {r_req/r_prose:.3f}")
    print(f"        kappa/(2(kappa-1)) = {k/(2*(k-1)):.4f}  (T_c/2 版と一致: "
          f"{abs(k/(2*(k-1)) - r_req/r_half) < 1e-9})")
    assert abs(k / (2 * (k - 1)) - r_req / r_half) < 1e-9
    print("  → この検定は kappa≈2（三角形波）からのずれを測っているだけで、")
    print("     生理学的な立ち上がり率の実在性を検定していない。")


# ---------------------------------------------------------------------------
# [D4] エリート必要RFDの増加は構成上の代数
# ---------------------------------------------------------------------------
def d4_elite_scaling_is_algebra() -> None:
    _hdr("[D4] 「エリートでは必要RFDが30-40%高い」は v の2次式の数値評価")

    kappa, beta, L_c, t_a, v = sp.symbols("kappa beta L_c t_a v", positive=True)
    # elite_recalibration.py: R(v) = kappa (1 + t_a v / L_c) v / (beta L_c)
    R_v = kappa * (1 + t_a * v / L_c) * v / (beta * L_c)
    R_expanded = sp.expand(R_v)
    print(f"  R(v) = {R_expanded}")
    print(f"  degree in v = {sp.degree(sp.Poly(R_expanded, v))}")
    assert sp.degree(sp.Poly(R_expanded, v)) == 2
    dR = sp.simplify(sp.diff(R_v, v))
    print(f"  dR/dv = {dR}  -> kappa,beta,L_c,t_a>0 のとき恒に正")
    assert sp.simplify(dR.subs({kappa: 1.74, beta: 0.425, L_c: 1.0, t_a: 0.128})) > 0

    k, b, Lc, ta = 1.7404, 0.4254, 1.0, 0.128
    def Rv(vv): return k * (1 + ta * vv / Lc) * vv / (b * Lc)
    r92, r115 = Rv(9.2), Rv(11.5)
    print(f"\n  数値: R(9.2) = {r92:.1f} Wb/s, R(11.5) = {r115:.1f} Wb/s"
          f"  -> +{(r115/r92-1)*100:.0f}%")
    print("  kappa, beta, L_c, t_a を速度非依存に固定した時点で単調増加は確定しており、")
    print("  経験的内容は「これら4量が速度によらない」という仮定のみ。")


# ---------------------------------------------------------------------------
# [D5] 縮退していない検証可能内容と、out-of-sample 予測の設計
# ---------------------------------------------------------------------------
def d5_what_is_testable(A_sym: sp.Expr) -> None:
    _hdr("[D5] 真に検証可能な内容と out-of-sample 予測の設計")

    print("  容量側 (rate-limited):  F_avail(t_c) = A t_c,      A = beta(1-beta) R")
    print("  要求側 (impulse balance): F_req(t_c) = mg (1 + t_a/t_c)")
    print("  v_max は両者の交点。A は較正点で A = F_avg^meas / T_c に固定される。\n")

    A = MEAS["F_avg"] / MEAS["T_c"]          # [Wb/s]
    print(f"  較正: A = F_avg/T_c = {A:.4f} Wb/s")

    def predict_vmax(t_a: float, L_c: float, A_: float = A) -> float:
        """A を凍結し、別コホートの (t_a, L_c) から v_max を予測する。"""
        # A t_c^2 - t_c - t_a = 0  (mg = 1 Wb)
        disc = 1.0 + 4.0 * A_ * t_a
        t_c = (1.0 + math.sqrt(disc)) / (2.0 * A_)
        return L_c / t_c

    v_self = predict_vmax(MEAS["T_aer"], MEAS["L_c"])
    print(f"  自己再現 (較正コホート): v_max = {v_self:.3f} m/s "
          f"(実測 {MEAS['v_top']:.2f}) -> 較正であって検証ではない")

    print("\n  out-of-sample 予測の感度（A 凍結、t_a と L_c のみ別コホート値）:")
    print(f"    {'t_a [s]':>8} {'L_c [m]':>8} {'v_max pred [m/s]':>18}")
    for t_a_, L_c_ in [(0.119, 0.98), (0.128, 1.00), (0.130, 1.05),
                       (0.115, 0.95), (0.125, 1.10)]:
        print(f"    {t_a_:>8.3f} {L_c_:>8.2f} {predict_vmax(t_a_, L_c_):>18.3f}")

    span = (predict_vmax(0.125, 1.10) - predict_vmax(0.115, 0.95))
    print(f"\n  上表の予測レンジ = {span:.2f} m/s。")
    print("  => (t_a, L_c) が実際に異なる独立コホートを与えれば、A 凍結の予測は")
    print("     自明でない散らばりを持つ。1:1 プロット（予測 vs 実測）で RMSE と")
    print("     バイアスを報告すれば、out-of-sample の検証として成立する。")
    print("  注意: A 自体が被験者間で不変という仮定が検証の対象であり、")
    print("        これが崩れる場合はそれ自体が報告すべき否定的結果である。")


def main() -> None:
    A = d1_symbolic_cancellation()
    d2_check2_is_check1()
    d3_circularity_ratio()
    d4_elite_scaling_is_algebra()
    d5_what_is_testable(A)

    _hdr("監査結果まとめ")
    print("  D1 成立: 較正後の力積収支は R, beta に依存しない（記号的に恒等）")
    print("  D2 成立: 検証2 は検証1 の言い換え。独立な検証ではない")
    print("  D3 成立: 非循環性の 20% は kappa と 2 の距離の言い換え")
    print("  D4 成立: エリート必要RFDの増加は構成上の代数")
    print("  D5     : 検証可能な内容は A の被験者間不変性。独立コホートで検定可能")
    print("\n  => 原稿の『3つの独立な検証』のうち、独立でも検証でもないものが2つ。")
    print("     残る経験的内容は F_peak(3.62) < F_ceiling(4.20) の1点のみで、")
    print("     これは Weyand 2010 の測定であって本モデルの予測ではない。")


if __name__ == "__main__":
    main()
