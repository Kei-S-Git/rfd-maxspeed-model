"""
図版スタイルの共通設定（Journal of Experimental Biology 準拠）。

JEB の figure 要件（manuscript preparation ページ）:
  - 最大寸法 180 mm x 210 mm
  - 解像度 300 ppi、ベクタ（EPS/PDF）が望ましい
  - カラーモード RGB
  - パネル記号は 12 pt 太字大文字（A, B, C）、その他ラベルは 8 pt Arial

投稿版の図版監査で見つかった既存図の欠陥（本モジュールで回避する）:
  - figsize 15 x 4.5 in を textwidth に縮小していたため軸ラベルが実効 4.7 pt
  - dpi=130 のラスタのみ
  - 軸ラベルの単位が "Wb"（weber の SI 記号）。体重比は "BW"
  - 色だけで系列を区別しておりグレースケール印刷で判別できない
"""

from __future__ import annotations

import os

MM = 1 / 25.4          # mm -> inch
MAX_W_MM = 180.0       # JEB の最大幅
DPI = 300              # JEB の最小解像度


def apply() -> None:
    """rcParams を JEB 準拠に設定する。matplotlib の import 後に呼ぶこと。"""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


def figsize(width_mm: float, height_mm: float) -> tuple[float, float]:
    """mm 指定で figsize を返す。幅が JEB の上限を超える場合は例外。"""
    if width_mm > MAX_W_MM:
        raise ValueError(f"width {width_mm} mm exceeds the JEB limit of {MAX_W_MM} mm")
    return (width_mm * MM, height_mm * MM)


def panel_letters(fig, axes, letters=None, dx=0.055, dy=0.045) -> None:
    """
    パネル記号（12 pt 太字大文字）を図座標で配置する。

    軸ラベルとの衝突を避けるため、tight_layout の後に図座標で置く。
    axes 座標に置くと、アスペクト比を固定したパネルで y 軸ラベルに重なる。
    """
    if letters is None:
        letters = [chr(ord("A") + i) for i in range(len(axes))]
    top = max(a.get_position().y1 for a in axes)
    for a, letter in zip(axes, letters):
        fig.text(a.get_position().x0 - dx, top + dy, letter,
                 fontsize=12, fontweight="bold", va="bottom", ha="left")


def save(fig, stem: str, repo_root: str | None = None) -> list[str]:
    """
    PNG (300 dpi) と PDF (ベクタ) を保存する。

    保存先は文脈で決める。output/ には常に書く。原稿ツリー（paper/）が存在する
    場合に限り paper/figures/ にも書き、原稿の図を更新する。公開用コード
    リリースには paper/ が無いので、そこに空のディレクトリを作らない。
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = ["output"]
    if os.path.isdir(os.path.join(repo_root, "paper")):
        targets.append(os.path.join("paper", "figures"))
    written = []
    for sub in targets:
        d = os.path.join(repo_root, sub)
        os.makedirs(d, exist_ok=True)
        png = os.path.join(d, stem + ".png")
        pdf = os.path.join(d, stem + ".pdf")
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf)
        written += [png, pdf]
    return written
