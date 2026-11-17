"""Builds docs/timeline.tex, the possession strips in the report:  python docs/make_figures.py

Compares the public 5-minute ground truth with our file and the official baseline. Only the
published ground truth and the two submission files are read; no match data is needed.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
ROWS = [
    ("Ground truth", GT),
    ("Our submission", os.path.join(ROOT, "evaluation", "results", "Ragnir.csv")),
    ("Official baseline", os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")),
]
FPS = 25
WIDTH_CM = 12.0
ROW_CM = 0.62
GAP_CM = 0.32
COLOURS = {"H": "colH", "A": "colA", "D": "colD", "X": "colD"}


def segments(labels):
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    return [(labels[s], s, e) for s, e in zip(starts, ends)]


def main():
    truth = pd.read_csv(GT)["label"].to_numpy()
    graded = np.flatnonzero(truth != "X")
    first, last = graded[0], graded[-1] + 1
    scale = WIDTH_CM / (last - first)

    out = []
    for row, (name, path) in enumerate(ROWS):
        labels = pd.read_csv(path)["label"].to_numpy()[first:last]
        y = -row * (ROW_CM + GAP_CM)
        out.append(r"\node[anchor=east, font=\small] at (-0.15,{:.3f}) {{{}}};".format(y + ROW_CM / 2, name))
        for label, s, e in segments(labels):
            out.append(r"\fill[{}] ({:.4f},{:.3f}) rectangle ({:.4f},{:.3f});".format(
                COLOURS[label], s * scale, y, e * scale, y + ROW_CM))
        n_poss = sum(1 for label, _, _ in segments(labels) if label in "HA")
        out.append(r"\node[anchor=west, font=\scriptsize, text=muted] at ({:.3f},{:.3f}) {{{} poss.}};".format(
            WIDTH_CM + 0.12, y + ROW_CM / 2, n_poss))

    bottom = -(len(ROWS) - 1) * (ROW_CM + GAP_CM) - 0.12
    seconds = (last - first) / FPS
    for t in range(0, int(seconds) + 1, 60):
        x = t * FPS * scale
        out.append(r"\draw[muted, line width=0.4pt] ({:.3f},{:.3f}) -- ({:.3f},{:.3f});".format(x, bottom, x, bottom - 0.1))
        out.append(r"\node[anchor=north, font=\scriptsize, text=muted] at ({:.3f},{:.3f}) {{{}:00}};".format(x, bottom - 0.1, t // 60))

    with open(os.path.join(ROOT, "docs", "timeline.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")
    print("wrote docs/timeline.tex")


if __name__ == "__main__":
    main()
