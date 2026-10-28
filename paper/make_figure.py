"""Fig. 1 of the ICALT draft: system pipeline. Run: python paper/make_figure.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 4.6})
fig, ax = plt.subplots(figsize=(3.45, 2.5), dpi=600)
ax.set_xlim(0, 100)
ax.set_ylim(0, 72)
ax.axis("off")

GREY, BLUE, GREEN = "#eeeeee", "#dce8f5", "#e3f1e0"


def box(x, y, w, h, text, fc=GREY, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="black", lw=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color="black", mutation_scale=5))


# Row 1: input
box(32, 64, 36, 6, "Classroom lecture recording\n(single fixed camera, ~1 h)", bold=True)
# Row 2: extractors
box(1, 50, 30, 8, "Screen-cropped OCR\n(EasyOCR, auto screen\ndetection)")
box(35, 50, 30, 8, "Speech recognition\n(Whisper + hallucination\nfiltering)")
box(69, 50, 30, 8, "Frame embedding\n(CLIP ViT-B/32)\n+ optical flow")
for x in (16, 50, 84):
    arrow(50, 64, x, 58.6)
# Row 3: per-step signals
box(10, 38, 80, 6, "Per-step stream (3 s): slide-text change, topic drift, visual change, CLIP")
for x in (16, 50, 84):
    arrow(x, 50, x, 44.6)
# Row 4: stage 1
box(10, 26, 80, 7, "Stage 1: Pedagogical Boundary Detector\n(temporal encoder, margin + segment InfoNCE)", fc=BLUE, bold=False)
arrow(50, 38, 50, 33.6)
# Row 5: stage 2
box(10, 13, 80, 8, "Stage 2: per segment [OCR | ASR | visual] tokens\n→ cross-modal fusion encoder → z_seg (InfoNCE with query projector)", fc=BLUE)
arrow(50, 26, 50, 21.6)
# Row 6: retrieval
box(1, 0.5, 30, 7, "Student query\n(Sentence-BERT)", fc=GREEN)
box(35, 0.5, 64, 7, "Ranked moments + jump links\nvs. BM25, dense (MiniLM, BGE), RRF hybrid, re-ranker", fc=GREEN)
arrow(31.4, 4, 34.6, 4)
arrow(50, 13, 60, 7.9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_pipeline.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("saved", out)
