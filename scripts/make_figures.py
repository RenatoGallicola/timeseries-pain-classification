#!/usr/bin/env python3
"""Render the README figures from the committed out-of-fold logits.

    python scripts/make_figures.py

Writes light and dark variants to assets/. Needs only numpy, scikit-learn and
matplotlib — no dataset, no GPU, no trained weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ENSEMBLE, LABELS  # noqa: E402

ASSETS = ROOT / "assets"

# --- Theme ----------------------------------------------------------------
# Single accent hue plus a de-emphasis gray: the "emphasis" form, where one
# series is the point and the rest are context. Both accent steps pass the
# lightness band, chroma floor and 3:1 contrast gate on their own surface.
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", accent="#2a78d6", context="#898781",
        # sequential blue, light -> dark; the lightest step is "near zero" and
        # is meant to recede toward the surface (the heatmap exception).
        ramp=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", accent="#3987e5", context="#898781",
        # reversed for the dark surface: near-zero recedes into the background
        ramp=["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
    ),
}

FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def _style(theme):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["ink"],
        "axes.labelcolor": theme["secondary"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["secondary"],
        "axes.edgecolor": theme["axis"],
    })


def load_oof():
    d = ROOT / "artifacts" / "oof_logits"
    return (np.load(d / "oof_logits_A.npy"), np.load(d / "oof_logits_B.npy"),
            np.load(d / "oof_logits_C.npy"), np.load(d / "oof_labels.npy"))


# --- Figure 1: ablation ---------------------------------------------------
def figure_ablation(mode: str) -> Path:
    """Dot plot, not a bar chart: the scores span 0.88-0.95, so bars from zero
    would look identical and bars from 0.86 would overstate the differences.
    Dots carry no area, so a non-zero axis is honest."""
    theme = THEMES[mode]
    _style(theme)
    a, b, c, y = load_oof()
    w = ENSEMBLE

    # The shipped configuration is A+B+C with the gate; the gate leaves the
    # out-of-fold predictions unchanged, so it shares this row rather than
    # sitting in a duplicate one. docs/RESULTS.md lists the two separately.
    rows = [
        ("Model C alone  ·  InceptionTime", c.argmax(1), False),
        ("Model B alone  ·  CNN-GRU, window 200", b.argmax(1), False),
        ("Model A alone  ·  CNN-GRU, window 96", a.argmax(1), False),
        ("A + B  ·  equal weights", (0.5 * a + 0.5 * b).argmax(1), False),
        ("A + B + C  ·  CMA-ES weights, submitted",
         (w.w_a * a + w.w_b * b + w.w_c * c).argmax(1), True),
    ]
    labels = [r[0] for r in rows]
    scores = [f1_score(y, r[1], average="weighted") for r in rows]
    is_final = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ypos = np.arange(len(rows))
    lo, hi = 0.868, 0.966

    for i, (score, final) in enumerate(zip(scores, is_final)):
        colour = theme["accent"] if final else theme["context"]
        # leader line from the axis start to the dot
        ax.plot([lo, score], [i, i], color=theme["grid"], lw=1.0, zorder=1,
                solid_capstyle="butt")
        ax.plot(score, i, "o", ms=11 if final else 9, color=colour, zorder=3,
                markeredgecolor=theme["surface"], markeredgewidth=2)
        ax.text(score + 0.0035, i, f"{score:.4f}", va="center", ha="left",
                fontsize=10.5, color=theme["ink"] if final else theme["secondary"],
                fontweight="600" if final else "normal", zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10.5)
    for tick, final in zip(ax.get_yticklabels(), is_final):
        tick.set_color(theme["ink"] if final else theme["secondary"])
        if final:
            tick.set_fontweight("600")

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(np.arange(0.88, 0.961, 0.02))
    ax.set_xlabel("Out-of-fold weighted F1   ·   661 subjects   ·   axis starts at 0.868",
                  fontsize=9.5, labelpad=9)

    ax.xaxis.grid(True, color=theme["grid"], lw=0.8, zorder=0)  # solid hairline
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0)

    # Centred on the figure rather than on the axes: the category labels push
    # the plot area to the right, so an axes-centred title reads as off-centre.
    fig.suptitle("Out-of-fold weighted F1 by ensemble configuration",
                 fontsize=13, fontweight="600", color=theme["ink"], y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / f"ablation-{mode}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Figure 2: error diversity --------------------------------------------
def figure_diversity(mode: str) -> Path:
    """Why an ensemble of three mediocre-to-good models beats the best of them.

    A real bar chart this time: these are counts of subjects, zero is a
    meaningful baseline, so bars from zero are honest.
    """
    theme = THEMES[mode]
    _style(theme)
    a, b, c, y = load_oof()

    wrong = {"A": a.argmax(1) != y, "B": b.argmax(1) != y, "C": c.argmax(1) != y}
    regions = [
        ("C only", (~wrong["A"]) & (~wrong["B"]) & wrong["C"], False),
        ("B only", (~wrong["A"]) & wrong["B"] & (~wrong["C"]), False),
        ("B and C", (~wrong["A"]) & wrong["B"] & wrong["C"], False),
        ("A and C", wrong["A"] & (~wrong["B"]) & wrong["C"], False),
        ("A only", wrong["A"] & (~wrong["B"]) & (~wrong["C"]), False),
        ("A and B", wrong["A"] & wrong["B"] & (~wrong["C"]), False),
        ("all three", wrong["A"] & wrong["B"] & wrong["C"], True),
    ]
    labels = [r[0] for r in regions]
    counts = [int(r[1].sum()) for r in regions]
    is_core = [r[2] for r in regions]

    order = sorted(range(len(counts)), key=lambda i: counts[i])
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    is_core = [is_core[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ypos = np.arange(len(counts))

    for i, (count, corey) in enumerate(zip(counts, is_core)):
        ax.barh(i, count, height=0.44, zorder=2,
                color=theme["accent"] if corey else theme["context"])
        ax.text(count + 0.8, i, str(count), va="center", ha="left", fontsize=10.5,
                color=theme["ink"] if corey else theme["secondary"],
                fontweight="600" if corey else "normal", zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10.5)
    for tick, corey in zip(ax.get_yticklabels(), is_core):
        tick.set_color(theme["ink"] if corey else theme["secondary"])
        if corey:
            tick.set_fontweight("600")

    ax.set_xlim(0, max(counts) * 1.13)
    ax.set_ylim(-0.7, len(counts) - 0.3)
    ax.set_xlabel("Subjects predicted incorrectly, out of fold   ·   661 total",
                  fontsize=9.5, labelpad=9)

    ax.xaxis.grid(True, color=theme["grid"], lw=0.8, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0)

    # Centred on the figure rather than on the axes: the category labels push
    # the plot area to the right, so an axes-centred title reads as off-centre.
    fig.suptitle("Misclassified subjects, grouped by which models miss them",
                 fontsize=13, fontweight="600", color=theme["ink"], y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / f"error-diversity-{mode}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Figure 3: confusion matrix -------------------------------------------
def figure_confusion(mode: str) -> Path:
    """Heatmap = sequential magnitude, one hue light->dark. Every cell also
    carries its count and row share, so nothing is encoded by colour alone."""
    theme = THEMES[mode]
    _style(theme)
    a, b, c, y = load_oof()
    w = ENSEMBLE
    preds = (w.w_a * a + w.w_b * b + w.w_c * c).argmax(1)

    counts = confusion_matrix(y, preds)
    shares = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", theme["ramp"])

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)

    names = [n.replace("_", " ") for n in LABELS]
    ax.set_xticks(range(3), names, fontsize=10.5, color=theme["secondary"])
    ax.set_yticks(range(3), names, fontsize=10.5, color=theme["secondary"])
    ax.set_xlabel("predicted", fontsize=10, labelpad=9)
    ax.set_ylabel("true", fontsize=10, labelpad=9)

    for i in range(3):
        for j in range(3):
            # Pick the ink from the cell's actual luminance, not from its value:
            # the ramp is reversed in dark mode, so a high share is a *light*
            # cell there and a dark one in light mode.
            red, green, blue, _ = cmap(shares[i, j])
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            on_light = luminance > 0.5
            strong = "#0b0b0b" if on_light else "#ffffff"
            soft = "#3d3d3b" if on_light else "#dcdcd6"
            ax.text(j, i - 0.11, f"{counts[i, j]}", ha="center", va="center",
                    fontsize=17, fontweight="600", color=strong)
            ax.text(j, i + 0.19, f"{shares[i, j] * 100:.1f}%", ha="center",
                    va="center", fontsize=9.5, color=soft)

    # 2px surface gap between cells instead of borders around them.
    # Interior boundaries only: drawing the outer edges too leaves a stray
    # hairline along the right and bottom of the grid.
    ax.set_xticks(np.arange(0.5, 2.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 2.5, 1), minor=True)
    ax.grid(which="minor", color=theme["surface"], linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.04)
    cbar.set_label("share of the true class", fontsize=9, color=theme["secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8.5, colors=theme["muted"])

    # Centred on the figure rather than on the axes: the category labels push
    # the plot area to the right, so an axes-centred title reads as off-centre.
    fig.suptitle("Confusion matrix of the final ensemble, out of fold",
                 fontsize=12.5, fontweight="600", color=theme["ink"], y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / f"confusion-matrix-{mode}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    for mode in ("light", "dark"):
        for path in (figure_ablation(mode), figure_diversity(mode),
                     figure_confusion(mode)):
            print(f"wrote {path.relative_to(ROOT)}  "
                  f"({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
