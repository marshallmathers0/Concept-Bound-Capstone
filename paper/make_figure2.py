"""Fig. 2 of the ICALT draft: per-step signals of one lecture with pilot query spans.
Run: python paper/make_figure2.py [video_id]"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PATHS
from src.utils.io_utils import load_features

VID = sys.argv[1] if len(sys.argv) > 1 else "college_lec03"
f = load_features(os.path.join(PATHS.features_dir, f"{VID}.npz"))
t = f["timestamps"] / 60.0


def smooth(x, w=5):
    x = x.astype(np.float32)
    y = np.convolve(x, np.ones(w) / w, mode="same")
    return (y - y.min()) / (y.max() - y.min() + 1e-8)


plt.rcParams.update({"font.family": "Times New Roman", "font.size": 6.5})
fig, axes = plt.subplots(3, 1, figsize=(3.45, 1.9), dpi=600, sharex=True)
series = [("Slide-text change", f["ocr_signal"]), ("Topic drift", f["topic_drift_signal"]),
          ("Visual change", f["visual_signal"])]

gold_path = os.path.join(PATHS.gold_dir, f"{VID}.A1.json")
spans = [(q["start_time"] / 60, q["end_time"] / 60) for q in json.load(open(gold_path))["queries"]] \
    if os.path.exists(gold_path) else []

for ax, (name, sig) in zip(axes, series):
    for k, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color="#f2c14e" if k % 2 == 0 else "#f7dc96", alpha=0.6, lw=0)
    ax.plot(t, smooth(sig), color="black", lw=0.5)
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=6)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(width=0.4, length=2)
axes[-1].set_xlabel("Lecture time (min)")
axes[-1].set_xlim(0, t[-1])
fig.tight_layout(h_pad=0.2)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_signals.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("saved", out, "spans:", len(spans))
