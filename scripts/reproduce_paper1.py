"""
論文1（RFD と最高速度）の全結果・全図を再現する。

JEB Theory & Modelling は "clear and free access to code and data" を要求する。
本スクリプトは、原稿に現れるすべての数値と図を、1コマンドで先頭から再生成する。

実行:
    python scripts/reproduce_paper1.py            # 全実行
    python scripts/reproduce_paper1.py --check    # 主張の機械検証のみ（図を描かない）

生成物:
    output/ と paper/figures/ に 5 点の図（PNG 300 dpi + ベクタ PDF）

データについて:
    本研究は新規測定を行っていない。使用した実測値はすべて公表論文の表からの
    転記であり、出典・n・役割（較正 / 検証）は src/external_validation.py の
    COHORTS に、較正元の値は src/calibrate_parameters.py の Measured に、
    それぞれソース内で明示されている。外部データファイルへの依存はない。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (モジュール, 説明, 図を出力するか)
STEPS = [
    ("rfd_maxspeed_model",
     "閉形式 v_max(R) の導出、数値根との照合、sympy による独立検証", True),
    ("calibrate_parameters",
     "Weyand 2010 による較正と力の予算 [Fig. 2]", True),
    ("elite_recalibration",
     "Weyand 2000 への外挿と必要 RFD [Fig. 3]", True),
    ("degeneracy_audit",
     "縮退監査 D1-D5（記号計算、全命題 assert）", False),
    ("external_validation",
     "独立4研究室による out-of-sample 検定 [Fig. 4]", True),
    ("robustness_analyses",
     "感度と波形非対称性 [Fig. 5]", True),
    ("figure_scaling_law",
     "較正パラメータによるトップスピード則 [Fig. 1]", True),
]

FIGURES = [
    "rfd_maxspeed_scaling", "calibrated_maxspeed", "elite_recalibration",
    "external_validation", "robustness_checks",
]


def run(module: str, quiet: bool) -> tuple[bool, float, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=SRC)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(SRC, module + ".py")],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    dt = time.time() - t0
    if not quiet and proc.stdout:
        print(proc.stdout.rstrip())
    tail = (proc.stderr or "").strip().splitlines()
    return proc.returncode == 0, dt, (tail[-1] if tail else "")


def main() -> int:
    check_only = "--check" in sys.argv
    quiet = check_only or "--quiet" in sys.argv

    steps = [s for s in STEPS if not (check_only and s[2])]
    print("=" * 74)
    print("論文1 再現実行" + ("（機械検証のみ）" if check_only else ""))
    print(f"  python {sys.version.split()[0]}  root={ROOT}")
    print("=" * 74)

    failed = []
    for module, desc, _ in steps:
        ok, dt, err = run(module, quiet)
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {module:<22} {dt:5.1f}s  {desc}")
        if not ok:
            failed.append((module, err))

    if not check_only:
        print("-" * 74)
        # 原稿ツリーがある場合のみ paper/figures/ も検査する（公開用リリースには無い）
        subs = ["output"]
        if os.path.isdir(os.path.join(ROOT, "paper")):
            subs.append(os.path.join("paper", "figures"))
        missing = []
        for stem in FIGURES:
            for sub in subs:
                for ext in (".png", ".pdf"):
                    p = os.path.join(ROOT, sub, stem + ext)
                    if not os.path.exists(p):
                        missing.append(os.path.relpath(p, ROOT))
        if missing:
            print(f"[FAIL] 生成されなかった図: {len(missing)} 件")
            for m in missing[:10]:
                print("       ", m)
            failed.append(("figures", f"{len(missing)} missing"))
        else:
            where = " と ".join(s.replace(os.sep, "/") + "/" for s in subs)
            print(f"[OK  ] 図 {len(FIGURES)} 点を {where}に PNG(300dpi)+PDF で出力")

    print("=" * 74)
    if failed:
        print("失敗:")
        for m, e in failed:
            print(f"  {m}: {e}")
        return 1
    print("すべて成功。原稿の数値・図はこの実行から再現される。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
